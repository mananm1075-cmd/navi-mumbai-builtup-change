"""Method B: supervised Random Forest classification, trained separately for every year
on temporally stable training areas.  Accuracy is measured on the held-out half of each
training rectangle (spatial hold-out)."""
import glob, json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from scipy import ndimage as nd
import s2lib as L, features as F
from training_areas import AREAS

RNG = np.random.default_rng(0)
N_PER_CLASS = 4000

def split_samples(X, year=None):
    from training_areas import YEAR_AREAS
    areas = {c: list(r) for c, r in AREAS.items()}
    for c, r in YEAR_AREAS.get(int(year), {}).items() if year else []:
        areas[c] = areas[c] + r
    tr, te = {}, {}
    for c, rects in areas.items():
        a, b = [], []
        for r0, r1, c0, c1 in rects:
            mid = (c0 + c1) // 2                       # left half -> train, right half -> test
            a.append(X[r0:r1, c0:mid].reshape(-1, X.shape[-1]))
            b.append(X[r0:r1, mid:c1].reshape(-1, X.shape[-1]))
        tr[c], te[c] = np.concatenate(a), np.concatenate(b)
    return tr, te

def stack(d, n=None):
    Xs, ys = [], []
    for c, x in d.items():
        if n and len(x) > n:
            x = x[RNG.choice(len(x), n, replace=False)]
        Xs.append(x); ys.append(np.full(len(x), c))
    return np.concatenate(Xs), np.concatenate(ys)

def mode_filter3(cls, n_classes=4):
    """3x3 majority filter on the class map (removes isolated misclassified pixels)."""
    counts = np.stack([nd.uniform_filter((cls == k).astype(np.float32), 3) for k in range(n_classes)])
    return counts.argmax(0).astype(np.uint8)

def run_year(path):
    hdr, bands, valid = L.load_year(path)
    X = F.build_features(bands)
    tr, te = split_samples(X)
    Xtr, ytr = stack(tr, N_PER_CLASS)
    Xte, yte = stack(te, N_PER_CLASS)
    rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=0)
    rf.fit(Xtr, ytr)
    pte = rf.predict(Xte)
    cm = confusion_matrix(yte, pte, labels=[0, 1, 2, 3])
    cls = rf.predict(X.reshape(-1, X.shape[-1])).reshape(X.shape[:2]).astype(np.uint8)
    cls_f = mode_filter3(cls)
    cls_f[~valid] = 255
    acc = {
        "overall": float(np.trace(cm) / cm.sum()),
        "kappa": float(cohen_kappa_score(yte, pte)),
        "builtup_producers": float(cm[0, 0] / cm[0].sum()),
        "builtup_users": float(cm[0, 0] / cm[:, 0].sum()),
        "cm": cm.tolist(),
        "importance": dict(zip(F.FEATURE_NAMES, rf.feature_importances_.round(3).tolist())),
    }
    return hdr, cls_f, valid, acc

if __name__ == "__main__":
    out = {}
    for p in sorted(glob.glob("data/*.s2clip.gz")):
        hdr, cls, valid, acc = run_year(p)
        y = hdr["date"][:4]
        np.savez_compressed(f"results/class_{y}.npz", cls=cls, valid=valid)
        out[y] = {"date": hdr["date"], "item": hdr["item_id"], "valid_km2": float(valid.sum() * 1e-4),
                  "builtup_km2": float((cls == 0).sum() * 1e-4), **acc}
        print(y, hdr["date"], f"built-up {out[y]['builtup_km2']:.2f} km2", f"OA {acc['overall']:.3f} kappa {acc['kappa']:.3f}",
              f"BU PA {acc['builtup_producers']:.3f} UA {acc['builtup_users']:.3f}", flush=True)
    json.dump(out, open("results/method_b_summary.json", "w"), indent=1)
