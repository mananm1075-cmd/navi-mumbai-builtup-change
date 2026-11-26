"""Per-pixel features for classification (spectral bands, indices, texture)."""
import numpy as np
from scipy import ndimage as nd
import s2lib as L

FEATURE_NAMES = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "NDBI", "MNDWI", "BSI",
                 "SWIR_ratio", "NIR_std5", "NIR_contrast5"]

def local_std(x, size=5):
    """Local standard deviation (variance texture, Lecture 18-19)."""
    m = nd.uniform_filter(x, size)
    m2 = nd.uniform_filter(x * x, size)
    return np.sqrt(np.maximum(m2 - m * m, 0))

def local_contrast(x, size=5):
    """Unser difference-histogram contrast: window mean of squared neighbour differences
    for displacements (0,1) and (1,0) (Lecture 18-19, sum & difference histograms)."""
    dh = np.zeros_like(x); dv = np.zeros_like(x)
    dh[:, :-1] = (x[:, 1:] - x[:, :-1]) ** 2
    dv[:-1, :] = (x[1:, :] - x[:-1, :]) ** 2
    return nd.uniform_filter((dh + dv) / 2, size)

def build_features(bands):
    b = {k: np.nan_to_num(v.astype(np.float32), nan=0.0) for k, v in bands.items() if k != "SCL"}
    ix = L.indices(b)
    feats = [b["B02"], b["B03"], b["B04"], b["B08"], b["B11"], b["B12"],
             ix["NDVI"], ix["NDBI"], ix["MNDWI"],
             L.nd(b["B11"] + b["B04"], b["B08"] + b["B02"]),
             b["B12"] / np.maximum(b["B11"], 1e-4),
             local_std(b["B08"]), local_contrast(b["B08"])]
    X = np.stack([np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0) for f in feats], axis=-1)
    return X.astype(np.float32)
