"""Method A: index-based built-up extraction (NDBI + NDVI + MNDWI rule), then morphological cleaning."""
import numpy as np
from scipy import ndimage as nd
from skimage.morphology import binary_opening, binary_closing, disk, remove_small_objects
import s2lib as L

def method_a(bands, valid, ndbi_thr=0.0, ndvi_thr=None, mndwi_thr=None):
    ix = L.indices(bands)
    mndwi_thr = L.otsu(ix["MNDWI"][valid]) if mndwi_thr is None else mndwi_thr
    water = (ix["MNDWI"] > mndwi_thr) & valid
    ndvi_thr = L.otsu(ix["NDVI"][valid & ~water]) if ndvi_thr is None else ndvi_thr
    raw = valid & ~water & (ix["NDVI"] < ndvi_thr) & (ix["NDBI"] > ndbi_thr)
    clean = binary_opening(raw, disk(1))          # remove isolated pixels (Lecture 15-17: opening)
    clean = binary_closing(clean, disk(1))        # fill small gaps (closing)
    clean = remove_small_objects(clean, max_size=8)  # drop specks < 9 px (0.09 ha)
    return {"raw": raw, "clean": clean, "water": water,
            "thr": dict(ndbi=ndbi_thr, ndvi=float(ndvi_thr), mndwi=float(mndwi_thr))}

def area_km2(mask):
    return mask.sum() * L.PIXEL_AREA_M2 / 1e6
