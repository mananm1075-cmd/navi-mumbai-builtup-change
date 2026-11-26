"""Method B (final): radiometric normalisation + one Random Forest for all years.

1. Relative radiometric normalisation: every band of every year is histogram-matched
   (quantile mapping) to the 2023 reference image.  Most of the scene does not change, so
   matching the distributions removes differences due to sun angle, haze and processing
   version (Lecture 20-21: relative correction; Lecture 14: histogram matching).
2. Features (bands, indices, texture) are computed from the normalised bands.
3. One Random Forest is trained on samples from the stable training areas of ALL years,
   so the same decision rule is applied to every year.
4. Accuracy: train on the left half of every training rectangle, test on the right half.
"""
import glob, json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import s2lib as L, features as F
from classify import split_samples, stack, mode_filter3

REF = "2023"
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
QS = np.linspace(0.5, 99.5, 199)

def match_to(src, ref_q, valid):
    q = np.nanpercentile(src[valid], QS)
    return np.interp(src, q, ref_q, left=None, right=None).astype(np.float32)

def load_all():
    years = {}
    for p in sorted(glob.glob("data/*.s2clip.gz")):
        hdr, b, v = L.load_year(p)
        years[hdr["date"][:4]] = (hdr, b, v)
    _, rb, rv = years[REF]
    refq = {k: np.nanpercentile(rb[k][rv], QS) for k in BANDS}
    for y, (hdr, b, v) in years.items():
        for k in BANDS:
            b[k] = match_to(b[k], refq[k], v)
    return years

def main(n_per_class_year=1500, p_thr=0.5, seed=0, outdir="results"):
    import classify; classify.RNG = np.random.default_rng(seed)
    years = load_all()
    feats, Xtr, ytr, Xte, yte = {}, [], [], [], []
    for y, (hdr, b, v) in years.items():
        X = F.build_features(b); feats[y] = X
        tr, te = split_samples(X, y)
        a, c = stack(tr, n_per_class_year); Xtr.append(a); ytr.append(c)
        a, c = stack(te, n_per_class_year); Xte.append(a); yte.append(c)
    Xtr, ytr, Xte, yte = map(np.concatenate, (Xtr, ytr, Xte, yte))
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    rf.fit(Xtr, ytr)
    pte = rf.predict(Xte)
    cm = confusion_matrix(yte, pte, labels=[0, 1, 2, 3])
    acc = {"overall": float(np.trace(cm) / cm.sum()), "kappa": float(cohen_kappa_score(yte, pte)),
           "builtup_producers": float(cm[0, 0] / cm[0].sum()), "builtup_users": float(cm[0, 0] / cm[:, 0].sum()),
           "cm": cm.tolist(), "importance": dict(zip(F.FEATURE_NAMES, rf.feature_importances_.round(3).tolist()))}
    out = {"accuracy": acc, "years": {}}
    for y, X in feats.items():
        hdr, b, v = years[y]
        prob = rf.predict_proba(X.reshape(-1, X.shape[-1])).reshape(*X.shape[:2], 4).astype(np.float32)
        cls = prob.argmax(-1).astype(np.uint8)
        cls[(cls == 0) & (prob[..., 0] < p_thr)] = prob[..., 1:].argmax(-1)[(cls == 0) & (prob[..., 0] < p_thr)] + 1
        cls = mode_filter3(cls)
        cls[~v] = 255
        np.savez_compressed(f"{outdir}/pooled_class_{y}.npz", cls=cls, valid=v, p_built=prob[..., 0])
        out["years"][y] = {"date": hdr["date"], "item": hdr["item_id"], "builtup_km2": float((cls == 0).sum() * 1e-4),
                           "valid_km2": float(v.sum() * 1e-4)}
        print(y, f"{out['years'][y]['builtup_km2']:.2f} km2", flush=True)
    print("OA", round(acc["overall"], 3), "kappa", round(acc["kappa"], 3), "BU PA/UA", round(acc["builtup_producers"], 3), round(acc["builtup_users"], 3))
    print(cm)
    json.dump(out, open(f"{outdir}/pooled_summary.json", "w"), indent=1)

if __name__ == "__main__":
    main()
