"""Split an .s2bundle (several .s2clip.gz files saved from the browser in one download)."""
import struct, sys
from pathlib import Path

def unbundle(path, out_dir="data"):
    raw = Path(path).read_bytes()
    assert raw[:10] == b"S2BUNDLE1\n", "not an s2bundle"
    pos, names = 10, []
    out = Path(out_dir); out.mkdir(exist_ok=True)
    while pos < len(raw):
        (n,) = struct.unpack("<I", raw[pos:pos + 4]); pos += 4
        name = raw[pos:pos + n].decode(); pos += n
        (size,) = struct.unpack("<I", raw[pos:pos + 4]); pos += 4
        (out / name).write_bytes(raw[pos:pos + size]); pos += size
        names.append(name)
    return names

if __name__ == "__main__":
    print(unbundle(sys.argv[1]))
