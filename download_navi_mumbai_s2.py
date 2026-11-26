"""
Download Sentinel-2 L2A clips of the Navi Mumbai study area for 10 winters (2017-2026).

Source: Element 84 Earth Search STAC catalogue (public, no login) -> Cloud-Optimized
GeoTIFFs on AWS. Only the pixels inside the study box are read, so each year is ~5-15 MB.

Output (one folder per year, in ./navi_mumbai_s2/):
    NM_<date>_10m.tif   bands: B02 (blue), B03 (green), B04 (red), B08 (NIR)   -- 10 m
    NM_<date>_20m.tif   bands: B11 (SWIR1), B12 (SWIR2), SCL (scene class)     -- 20 m
    NM_<date>_meta.json item id, collection, processing baseline, scale/offset

Values are saved exactly as stored (raw digital numbers); conversion to reflectance
is done later in the processing notebook using the metadata.

Setup (once):
    pip install requests rasterio pyproj
Run:
    python download_navi_mumbai_s2.py
Then zip the "navi_mumbai_s2" folder and upload it.
"""
import json
import os

import rasterio
import requests
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

STAC = "https://earth-search.aws.element84.com/v1/search"
BBOX_LL = (72.98, 18.93, 73.15, 19.06)  # lon_min, lat_min, lon_max, lat_max
TILE = "43QBB"
DATES = {  # chosen dates (main, backup)
    2017: ("2017-02-12", "2017-03-04"),
    2018: ("2018-02-12", "2018-02-17"),
    2019: ("2019-02-17", "2019-03-09"),
    2020: ("2020-02-27", "2020-03-08"),
    2021: ("2021-03-03", "2021-02-26"),
    2022: ("2022-02-21", "2022-02-16"),
    2023: ("2023-02-11", "2023-02-16"),
    2024: ("2024-02-21", "2024-03-07"),
    2025: ("2025-02-20", "2025-03-07"),
    2026: ("2026-03-07", "2026-03-09"),
}
COLLECTIONS = ["sentinel-2-c1-l2a", "sentinel-2-l2a"]  # prefer Collection-1
GROUPS = {
    "10m": [("blue", "B02"), ("green", "B03"), ("red", "B04"), ("nir", "B08")],
    "20m": [("swir16", "B11"), ("swir22", "B12"), ("scl", "SCL")],
}
OUT = "navi_mumbai_s2"

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")


def find_item(date):
    """Return (collection, item) for tile T43QBB on the given date, preferring Collection-1."""
    for col in COLLECTIONS:
        body = {
            "collections": [col],
            "bbox": list(BBOX_LL),
            "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z",
            "limit": 20,
        }
        r = requests.post(STAC, json=body, timeout=60)
        r.raise_for_status()
        for f in r.json().get("features", []):
            if TILE in f["id"]:
                return col, f
    return None, None


def read_clip(href):
    """Read only the study-box window from a COG; return array, profile."""
    with rasterio.open(href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *BBOX_LL, densify_pts=21)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        data = src.read(1, window=win)
        prof = src.profile.copy()
        prof.update(
            driver="GTiff",
            width=data.shape[1],
            height=data.shape[0],
            transform=src.window_transform(win),
            compress="deflate",
            tiled=False,
        )
        prof.pop("blockxsize", None)
        prof.pop("blockysize", None)
    return data, prof


def save_year(year):
    for date in DATES[year]:
        col, item = find_item(date)
        if item:
            break
        print(f"  {date}: not found, trying backup")
    if not item:
        print(f"{year}: NO IMAGE FOUND")
        return
    folder = os.path.join(OUT, str(year))
    os.makedirs(folder, exist_ok=True)
    meta = {
        "year": year,
        "date": date,
        "item_id": item["id"],
        "collection": col,
        "properties": item["properties"],
        "bands": {},
    }
    for res, bands in GROUPS.items():
        arrays, prof = [], None
        for key, name in bands:
            asset = item["assets"][key]
            data, p = read_clip(asset["href"])
            prof = prof or p
            arrays.append(data)
            meta["bands"][name] = {
                "asset": key,
                "href": asset["href"],
                "raster:bands": asset.get("raster:bands"),
            }
        prof.update(count=len(arrays), dtype=arrays[0].dtype)
        path = os.path.join(folder, f"NM_{date}_{res}.tif")
        with rasterio.open(path, "w", **prof) as dst:
            for i, (arr, (_, name)) in enumerate(zip(arrays, bands), start=1):
                dst.write(arr, i)
                dst.set_band_description(i, name)
        print(f"  saved {path}  {arrays[0].shape}")
    with open(os.path.join(folder, f"NM_{date}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    print(f"{year}: {date}  {item['id']}  ({col})")


if __name__ == "__main__":
    for y in sorted(DATES):
        try:
            save_year(y)
        except Exception as e:  # keep going if one year fails
            print(f"{y}: FAILED -> {e}")
    print("\nDone. Zip the folder", OUT, "and upload it.")
