"""Core processing for the Navi Mumbai built-up change project (GNR607).

Steps per year:
  1. read the .s2clip.gz package
  2. DN -> surface reflectance using the scale/offset stored with each band
  3. bring 20 m bands (B11, B12, SCL) onto the 10 m grid (each 20 m pixel -> 2x2)
  4. cloud / shadow / no-data mask from SCL
  5. spectral indices (NDVI, NDBI, MNDWI)
"""
import numpy as np

from read_s2clip import read_s2clip

PIXEL_AREA_M2 = 10 * 10
SCL_BAD = {0, 1, 3, 8, 9, 10}  # no-data, saturated, cloud shadow, cloud med/high, cirrus


def load_year(path):
    hdr, raw = read_s2clip(path)
    info = {b["name"]: b for b in hdr["bands"]}
    ref_shape = (info["B02"]["height"], info["B02"]["width"])
    t10 = info["B02"]["transform"]

    def up(a):  # 20 m -> 10 m by pixel replication (nearest neighbour, keeps values)
        a = np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)
        return a[: ref_shape[0], : ref_shape[1]]

    bands = {}
    for name, arr in raw.items():
        b = info[name]
        assert b["transform"][2] == t10[2] and b["transform"][5] == t10[5], f"{name} grid offset"
        if b["transform"][0] == 20:
            arr = up(arr)
        if name == "SCL":
            bands[name] = arr.astype(np.uint8)
            continue
        rb = (b.get("raster_bands") or [{}])[0] or {}
        scale = rb.get("scale", 1e-4)
        offset = rb.get("offset", 0.0)
        r = arr.astype(np.float32) * scale + offset
        r[arr == 0] = np.nan  # stored no-data
        bands[name] = r
    valid = ~np.isin(bands["SCL"], list(SCL_BAD))
    for k in ("B02", "B03", "B04", "B08", "B11", "B12"):
        valid &= np.isfinite(bands[k])
    return hdr, bands, valid


def nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def indices(bands):
    return {
        "NDVI": nd(bands["B08"], bands["B04"]),   # vegetation (Lecture 14)
        "NDBI": nd(bands["B11"], bands["B08"]),   # built-up
        "MNDWI": nd(bands["B03"], bands["B11"]),  # water (modified NDWI)
    }


def stretch(rgb, lo=2, hi=98, valid=None):
    """Percentile linear stretch to 0-1 for display."""
    out = np.zeros_like(rgb)
    for i in range(rgb.shape[-1]):
        c = rgb[..., i]
        v = c[valid] if valid is not None else c[np.isfinite(c)]
        a, b = np.nanpercentile(v, [lo, hi])
        out[..., i] = np.clip((c - a) / (b - a), 0, 1)
    return np.nan_to_num(out)


def otsu(x, bins=512):
    x = x[np.isfinite(x)]
    h, e = np.histogram(x, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w = np.cumsum(h)
    m = np.cumsum(h * c)
    wb, wf = w, w[-1] - w
    mb = m / np.maximum(wb, 1)
    mf = (m[-1] - m) / np.maximum(wf, 1)
    between = wb * wf * (mb - mf) ** 2
    return c[np.argmax(between)]
