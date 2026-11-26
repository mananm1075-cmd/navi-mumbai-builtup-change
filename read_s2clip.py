"""Read the .s2clip.gz packages saved from the browser and write GeoTIFFs.

Package layout (gzip-compressed):
    b"S2CLIP1\n" | uint32 little-endian header length | JSON header | band arrays (uint16, row-major)
"""
import gzip
import json
import struct
import sys
from pathlib import Path

import numpy as np


def read_s2clip(path):
    raw = gzip.decompress(Path(path).read_bytes())
    assert raw[:8] == b"S2CLIP1\n", "not an s2clip file"
    (hlen,) = struct.unpack("<I", raw[8:12])
    hdr = json.loads(raw[12:12 + hlen].decode())
    pos = 12 + hlen
    bands = {}
    for b in hdr["bands"]:
        n = b["width"] * b["height"]
        arr = np.frombuffer(raw, dtype="<u2", count=n, offset=pos).reshape(b["height"], b["width"])
        bands[b["name"]] = arr
        pos += 2 * n
    assert pos == len(raw), "size mismatch"
    return hdr, bands


def to_geotiffs(path, out_dir):
    import rasterio
    from rasterio.transform import Affine

    hdr, bands = read_s2clip(path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for b in hdr["bands"]:
        t = b["transform"]
        prof = dict(driver="GTiff", width=b["width"], height=b["height"], count=1,
                    dtype="uint16", crs=hdr["crs"], compress="deflate",
                    transform=Affine(t[0], t[1], t[2], t[3], t[4], t[5]))
        with rasterio.open(out / f"NM_{hdr['date']}_{b['name']}.tif", "w", **prof) as dst:
            dst.write(bands[b["name"]], 1)
    (out / f"NM_{hdr['date']}_meta.json").write_text(json.dumps(hdr, indent=1))
    return hdr


if __name__ == "__main__":
    for p in sys.argv[1:]:
        h = to_geotiffs(p, Path(p).name.split(".")[0])
        print(p, h["date"], h["item_id"], h["collection"])
