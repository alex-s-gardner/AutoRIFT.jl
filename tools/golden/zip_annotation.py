"""Read one annotation XML out of a remote Sentinel-1 SAFE zip, over HTTP range requests.

    python zipxml.py <granule> <swath> <pol>

A granule is ~8 GiB and its annotation is a few hundred kilobytes, so the whole archive is never
transferred: the End Of Central Directory record is read from the tail, the central directory from the
offset it names, and then only the one member. Authentication is `curl -n`, which takes the Earthdata
entry from `~/.netrc`.
"""
import io, re, struct, subprocess, sys, zlib

granule, swath, pol = sys.argv[1], sys.argv[2], sys.argv[3].lower()
mission = granule[:3]
url = f"https://datapool.asf.alaska.edu/SLC/S{mission[2]}/{granule}.zip"


def fetch(rng=None):
    cmd = ["curl", "-sfL", "-n", url]
    if rng:
        cmd += ["-r", rng]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"curl failed ({out.returncode}) for range {rng}: {out.stderr[:200]}")
    return out.stdout


def size():
    # A one-byte ranged GET, read for its `Content-Range` total. ASF refuses HEAD on the data pool, and
    # the redirect chain's own `Content-Length` describes the redirect rather than the object.
    out = subprocess.run(["curl", "-sL", "-n", "--max-redirs", "12", "-r", "-1", "-D", "-",
                          "-o", "/dev/null", url], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        m = re.match(r"(?i)content-range:\s*bytes\s+\d+-\d+/(\d+)", line.strip())
        if m:
            return int(m.group(1))
    raise RuntimeError("no Content-Range; headers were:\n" + out.stdout[-600:])


n = size()
print(f"{granule}.zip is {n / 2**30:.2f} GiB")

# The End Of Central Directory record, in the last 64 KiB.
tail = fetch(f"{n - 65536}-{n - 1}")
i = tail.rfind(b"PK\x05\x06")
if i < 0:
    raise RuntimeError("no EOCD found; the archive may be Zip64 beyond this tail")
cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
if cd_off == 0xFFFFFFFF:
    j = tail.rfind(b"PK\x06\x06")
    cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
print(f"central directory: {cd_size} bytes at {cd_off}")

cd = fetch(f"{cd_off}-{cd_off + cd_size - 1}")
want = re.compile(rf"annotation/s1[abc]-iw{swath}-slc-{pol}-.*\.xml$")
entry = None
p = 0
while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
    csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
    nlen, elen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
    lho, = struct.unpack("<I", cd[p + 42:p + 46])
    name = cd[p + 46:p + 46 + nlen].decode()
    if want.search(name):
        entry = (name, lho, csize, usize)
        break
    p += 46 + nlen + elen + clen
if entry is None:
    raise RuntimeError(f"no annotation matching IW{swath} {pol.upper()} in the central directory")
name, lho, csize, usize = entry
print(f"member {name}: {csize} compressed, {usize} raw, at {lho}")

# The local header, then the member's bytes.
head = fetch(f"{lho}-{lho + 29}")
method, = struct.unpack("<H", head[8:10])
nlen, elen = struct.unpack("<HH", head[26:30])
start = lho + 30 + nlen + elen
raw = fetch(f"{start}-{start + csize - 1}")
xml = zlib.decompress(raw, -15) if method == 8 else raw
print(f"fetched {len(raw) / 1024:.1f} KiB compressed -> {len(xml) / 1024:.1f} KiB of XML")

for tag in ("linesPerBurst", "samplesPerBurst", "slantRangeTime", "rangePixelSpacing"):
    m = re.search(rf"<{tag}>([^<]+)</{tag}>", xml.decode("utf-8", "replace"))
    print(f"  {tag:18s} {m.group(1) if m else '(absent)'}")
nb = len(re.findall(r"<burst>", xml.decode("utf-8", "replace")))
print(f"  bursts             {nb}")
