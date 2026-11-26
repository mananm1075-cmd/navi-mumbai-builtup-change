"""Build the final outputs from the classification results:
   results/builtup_area_by_year.csv, figures, change-year map, validation points."""
import csv, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pyproj import Transformer
import s2lib as L, method_a as A, trajectory as T

YEARS = list(range(2017, 2027))
OUT = "results"

# ---------- load ----------
P = np.stack([np.load(f"{OUT}/pooled_class_{y}.npz")["p_built"] for y in YEARS]).astype(np.float32)
CLS = np.stack([np.load(f"{OUT}/pooled_class_{y}.npz")["cls"] for y in YEARS])
summ = json.load(open(f"{OUT}/pooled_summary.json"))
dates = {int(y): v["date"] for y, v in summ["years"].items()}
raw_b = [float((CLS[i] == 0).sum() * 1e-4) for i in range(len(YEARS))]

tc, meth_a, hdr23 = {}, [], None
for p in sorted(glob.glob("data/*.s2clip.gz")):
    h, b, v = L.load_year(p)
    y = int(h["date"][:4])
    tc[y] = L.stretch(np.dstack([b["B04"], b["B03"], b["B02"]]), valid=v)
    meth_a.append(A.area_km2(A.method_a(b, v)["clean"]))
    if y == 2023:
        hdr23 = h

# ---------- trajectory (final series) ----------
r = T.analyse(P, YEARS)
area = r["area_km2"]
inc = [None] + [area[i] - area[i - 1] for i in range(1, len(YEARS))]
pct = [None] + [100 * inc[i] / area[i - 1] for i in range(1, len(YEARS))]

def label(p):
    if p is None:
        return ""
    if p >= 3.0:
        return "significant increase"
    if p <= 1.0:
        return "nearly unchanged"
    return "steady growth"

with open(f"{OUT}/builtup_area_by_year.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["year", "image_date", "builtup_km2_final", "change_km2", "change_pct", "assessment",
                "builtup_km2_raw_per_year_RF", "builtup_km2_methodA_index"])
    for i, y in enumerate(YEARS):
        w.writerow([y, dates[y], round(area[i], 2), "" if inc[i] is None else round(inc[i], 2),
                    "" if pct[i] is None else round(pct[i], 2), label(pct[i]), round(raw_b[i], 2), round(meth_a[i], 2)])

# ---------- Figure 1: the graph ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [3, 2]}, sharex=True)
ax1.plot(YEARS, area, "-o", color="#c0392b", lw=2.5, ms=7, label="Built-up area (Random Forest + temporal consistency)")
for i, y in enumerate(YEARS):
    ax1.annotate(f"{area[i]:.1f}", (y, area[i]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
ax1.set_ylabel("Built-up area (km²)")
ax1.set_title("Navi Mumbai (Ulwe–Kharghar–Panvel, 263 km²): built-up area vs year\nSentinel-2 L2A, one cloud-free image per year (11 Feb – 7 Mar)")
ax1.grid(alpha=.3)
ax1.set_ylim(min(area) - 4, max(area) + 4)
ax1.legend(loc="upper left", fontsize=9)
cols = {"significant increase": "#c0392b", "steady growth": "#e59866", "nearly unchanged": "#95a5a6"}
for i in range(1, len(YEARS)):
    ax2.bar(YEARS[i], inc[i], color=cols[label(pct[i])], width=.65)
    ax2.text(YEARS[i], inc[i] + .05, f"+{inc[i]:.2f}\n({pct[i]:.1f}%)", ha="center", va="bottom", fontsize=8)
ax2.set_ylabel("Increase from previous year (km²)")
ax2.set_ylim(0, max(inc[1:]) * 1.45)
ax2.grid(alpha=.3, axis="y")
from matplotlib.patches import Patch
ax2.legend(handles=[Patch(color=c, label=f"{k}") for k, c in cols.items()], fontsize=9, loc="upper left",
           title="≥3 %/yr significant, ≤1 %/yr nearly unchanged", title_fontsize=8)
ax2.set_xticks(YEARS)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_builtup_area_vs_year.png", dpi=150)
plt.close()

# ---------- Figure 2: methods compared ----------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(YEARS, meth_a, "-s", color="#7f8c8d", label="Method A: index rule (NDBI>0, NDVI, MNDWI)")
ax.plot(YEARS, raw_b, "-^", color="#2980b9", label="Method B: Random Forest, each year on its own")
ax.plot(YEARS, area, "-o", color="#c0392b", lw=2.5, label="Method B + temporal consistency (final)")
ax.set_ylabel("Built-up area (km²)"); ax.set_xticks(YEARS); ax.grid(alpha=.3); ax.legend(fontsize=9, loc="center right")
ax.set_title("Why the final series uses method B with temporal consistency")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_methods_compared.png", dpi=130); plt.close()

# ---------- Figure 3: change-year map ----------
g = np.repeat(tc[2026].mean(-1, keepdims=True) * 0.55, 3, -1)
g[r["stable"]] = [0.98, 0.76, 0.80]
cy = np.ma.masked_where(~r["new"], r["change_year"])
fig, ax = plt.subplots(1, 2, figsize=(20, 8.6))
ax[0].imshow(tc[2017]); ax[0].set_title(f"True colour {dates[2017]}")
ax[1].imshow(g)
im = ax[1].imshow(cy, cmap=plt.get_cmap("turbo", 9), vmin=2017.5, vmax=2026.5, interpolation="nearest")
cb = plt.colorbar(im, ax=ax[1], fraction=.035, ticks=YEARS[1:]); cb.set_label("Year first mapped as built-up")
ax[1].set_title("Pink: built-up throughout 2017–2026.  Colours: new built-up by year")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_change_year_map.png", dpi=110); plt.close()

# ---------- Figure 4: yearly maps ----------
fig, axs = plt.subplots(2, 5, figsize=(25, 9.2))
for i, (y, a) in enumerate(zip(YEARS, axs.flat)):
    built = r["stable"] | (r["new"] & (r["change_year"] <= y))
    img = tc[y].copy(); img[built] = 0.45 * img[built] + 0.55 * np.array([0.9, 0.1, 0.1])
    a.imshow(img); a.set_title(f"{dates[y]}  –  {area[i]:.1f} km²", fontsize=13); a.axis("off")
plt.suptitle("Built-up area (red) each year", fontsize=16)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_yearly_builtup_maps.png", dpi=80); plt.close()

# ---------- Figure 5: classification example ----------
cm = ListedColormap(["#e31a1c", "#33a02c", "#1f78b4", "#fdbf6f"])
fig, ax = plt.subplots(1, 2, figsize=(20, 8.4))
for a, y in zip(ax, (2017, 2026)):
    c = np.ma.masked_equal(CLS[YEARS.index(y)], 255)
    a.imshow(c, cmap=cm, vmin=0, vmax=3, interpolation="nearest"); a.set_title(f"Random Forest classes {dates[y]}"); a.axis("off")
ax[1].legend(handles=[Patch(color=cm(i), label=n) for i, n in enumerate(["Built-up", "Vegetation", "Water", "Bare / dry hills"])],
             loc="lower right", fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_classification_2017_2026.png", dpi=80); plt.close()

# ---------- validation points for Google Earth ----------
rng = np.random.default_rng(42)
tf = hdr23["bands"][0]["transform"]
to_ll = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
rows = []
for name, mask, n in (("new_builtup", r["new"], 40), ("stable_builtup", r["stable"], 30),
                      ("not_builtup", ~(r["new"] | r["stable"]), 30)):
    ys, xs = np.nonzero(mask)
    for k in rng.choice(len(ys), n, replace=False):
        yy, xx = ys[k], xs[k]
        lon, lat = to_ll.transform(tf[2] + (xx + .5) * tf[0], tf[5] + (yy + .5) * tf[4])
        rows.append([len(rows) + 1, name, int(r["change_year"][yy, xx]) or "", round(lat, 6), round(lon, 6)])
with open(f"{OUT}/validation_points.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["id", "mapped_as", "mapped_change_year", "lat", "lon", "your_check (yes/no)"])
    for row in rows: w.writerow(row + [""])
with open(f"{OUT}/validation_points.kml", "w") as fh:
    fh.write('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n')
    for i, name, yr, lat, lon in rows:
        fh.write(f"<Placemark><name>{i} {name} {yr}</name><Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>\n")
    fh.write("</Document></kml>\n")

json.dump({"years": YEARS, "area_km2": area, "increment_km2": inc, "increment_pct": pct,
           "raw_rf_km2": raw_b, "method_a_km2": meth_a, "stable_km2": float(r["stable"].sum() * 1e-4),
           "new_km2": float(r["new"].sum() * 1e-4), "accuracy": summ["accuracy"]},
          open(f"{OUT}/final_summary.json", "w"), indent=1)
print("done")
