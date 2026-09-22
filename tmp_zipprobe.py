# Scratch: does ASF's burst extractor report a different `samplesPerBurst` than the SAFE annotation?
#
# That is the one number blocking two of the five Sentinel-1 SLC cases: `merge_swaths` takes the mosaic's
# range width from the COMPASS CSLC raster of the far subswath's first burst, which equals
# `samplesPerBurst` on three cases and is 194 and 85 narrower on the other two.
#
# The granule is 4 GB and the answer is a few hundred bytes of XML. ASF's datapool honours range requests
# once `curl --netrc --cookie-jar` has done the URS redirect, and a zip's central directory is at the end,
# so the annotation is reachable with three ranged reads.

import io
import re
import subprocess
import sys
import zipfile

GRANULE = "S1A_IW_SLC__1SSH_20151120T080202_20151120T080229_008685_00C5A7_105E"
URL = f"https://datapool.asf.alaska.edu/SLC/SA/{GRANULE}.zip"
JAR = "/tmp/asf.jar"


def curl_range(first, last):
    """Bytes `first`..`last` inclusive, through the auth flow the extractor uses."""
    out = subprocess.run(
        ["curl", "--silent", "--show-error", "--location", "--netrc",
         "--cookie-jar", JAR, "--cookie", JAR, "--max-time", "300",
         "--range", f"{first}-{last}", URL],
        capture_output=True, check=True)
    return out.stdout


def content_length():
    out = subprocess.run(
        ["curl", "--silent", "--location", "--netrc", "--cookie-jar", JAR, "--cookie", JAR,
         "--range", "0-0", "--write-out", "%{size_download} %{http_code}", "--output", "/dev/null",
         "--head", URL],
        capture_output=True, text=True, check=True)
    # `--head` after a redirect does not always report a length, so fall back to a ranged GET whose
    # Content-Range carries the total.
    r = subprocess.run(
        ["curl", "--silent", "--location", "--netrc", "--cookie-jar", JAR, "--cookie", JAR,
         "--range", "0-0", "--dump-header", "-", "--output", "/dev/null", URL],
        capture_output=True, text=True, check=True)
    m = re.search(r"[Cc]ontent-[Rr]ange:\s*bytes\s+\d+-\d+/(\d+)", r.stdout)
    if not m:
        sys.exit(f"no Content-Range in:\n{r.stdout[:400]}")
    return int(m.group(1))


class RangeFile(io.RawIOBase):
    """A read-only file over HTTP ranges, enough for `zipfile` to work."""

    def __init__(self, size):
        self._size = size
        self._pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self._size + offset
        return self._pos

    def tell(self):
        return self._pos

    def readinto(self, b):
        n = len(b)
        if n == 0 or self._pos >= self._size:
            return 0
        last = min(self._pos + n, self._size) - 1
        data = curl_range(self._pos, last)
        b[: len(data)] = data
        self._pos += len(data)
        return len(data)


size = content_length()
print(f"granule is {size / 2**30:.2f} GiB")

with zipfile.ZipFile(io.BufferedReader(RangeFile(size), buffer_size=1 << 20)) as z:
    names = [n for n in z.namelist() if "/annotation/s1a-iw" in n and n.endswith(".xml")
             and "-slc-hh-" in n]
    print(f"annotation entries: {len(names)}")
    for n in sorted(names):
        xml = z.read(n).decode()
        spb = re.search(r"<samplesPerBurst>(\d+)</samplesPerBurst>", xml)
        lpb = re.search(r"<linesPerBurst>(\d+)</linesPerBurst>", xml)
        nb = len(re.findall(r"<burst>", xml))
        srt = re.search(r"<slantRangeTime>([\d.eE+-]+)</slantRangeTime>", xml)
        print(f"  {n.split('/')[-1]}: samplesPerBurst {spb.group(1)}  linesPerBurst {lpb.group(1)}"
              f"  bursts {nb}  slantRangeTime {srt.group(1) if srt else '?'}")
