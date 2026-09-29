"""Run the Python reference end to end on every golden case, one subprocess per case.

    micromamba run -n arift-ref python tools/ab/golden_python_all.py [--cap 3600] [--out FILE]

One process per case because `ru_maxrss` is a high-water mark: two cases in one process both report
the larger. `golden_python.py` does the measurement; this only discovers the cases, orders them, and
collects the `RESULT` lines.

**Ordered smallest grid first.** The reference is orders of magnitude slower than AutoRIFT.jl, so a
long run may not reach the end; taking the cheap cases first means a partial sweep still covers most
of the set rather than stalling on the first granule. `--cap` bounds each case so one that cannot
finish costs its cap rather than the whole run.

Cases with no captured inputs are reported as such rather than skipped silently — a missing capture is
a gap in the comparison, not an absence of one.
"""

import json
import os
import subprocess
import sys

RUNS = os.path.join(os.path.expanduser("~"), "data", "autorift", "tests", "golden_tests", "runs")
HERE = os.path.dirname(os.path.abspath(__file__))


def captures():
    """(case, capture-dir, grid points) for every case with captured inputs, smallest grid first."""
    out = []
    for case in sorted(os.listdir(RUNS)):
        base = os.path.join(RUNS, case)
        if not os.path.isdir(base):
            continue
        found = None
        for run in sorted(os.listdir(base), key=lambda s: (not s.isdigit(), s)):
            d = os.path.join(base, run, "capture")
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "call1.json")):
                found = d
                break
        if found is None:
            out.append((case, None, -1))
            continue
        try:
            m = json.load(open(os.path.join(found, "call1.json")))
            shape = m["arrays"]["in_xGrid"]["shape"]
            out.append((case, found, shape[0] * shape[1]))
        except Exception:
            out.append((case, found, 0))
    return sorted(out, key=lambda t: (t[2] < 0, t[2]))


def main():
    cap = sys.argv[sys.argv.index("--cap") + 1] if "--cap" in sys.argv else "3600"
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else \
        os.path.join(HERE, "golden_python.tsv")
    rows = captures()
    print("%d cases, %d with captures" % (len(rows), sum(1 for r in rows if r[1])))
    sys.stdout.flush()
    with open(out, "w") as fh:
        fh.write("case\tstatus\trunautorift_s\ttotal_s\tpeak_bytes\tmeasured\tgrid_rows\tgrid_cols\tcv2_threads\n")
        for i, (case, d, npts) in enumerate(rows, 1):
            if d is None:
                print("[%d/%d] %s  NO CAPTURE" % (i, len(rows), case[:58]))
                fh.write("%s\tno_capture\tnan\tnan\t0\t0\t0\t0\t0\n" % case)
                fh.flush()
                continue
            print("[%d/%d] %s  (%d grid points)" % (i, len(rows), case[:58], npts))
            sys.stdout.flush()
            r = subprocess.run(
                [sys.executable, os.path.join(HERE, "golden_python.py"), d, "--cap", cap],
                capture_output=True, text=True)
            line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), None)
            if line is None:
                tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["no output"]
                print("    FAILED: %s" % tail[0][:160])
                fh.write("%s\tharness_failed\tnan\tnan\t0\t0\t0\t0\t0\n" % case)
            else:
                f = line.split("\t")
                print("    %s  %.1f s  peak %.2f GiB  measured %s" % (
                    f[2], float(f[3]), int(f[5]) / 2**30, f[6]))
                fh.write("\t".join(f[1:]) + "\n")
            fh.flush()
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
