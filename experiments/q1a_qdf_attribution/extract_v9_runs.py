# Extract the small per-run files of protocol v9 from remote/q1a_v9_results.tar.gz into v9_pulled/runs_v9.
# The long result folder below res/ and ck/ is renamed to run/ (Windows path limit), as in the code package.
#   python extract_v9_runs.py
import os
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
TAR = os.path.join(HERE, "remote", "q1a_v9_results.tar.gz")
OUT = os.path.join(HERE, "v9_pulled")
KEEP = {"log.txt", "stdout.txt", "stub_report.json", "config.yaml", "metrics.npy", "sigma_target.json"}

n = 0
with tarfile.open(TAR, "r:gz") as t:
    for m in t:
        parts = m.name.split("/")
        if not m.isfile() or parts[0] != "runs_v9" or parts[-1] not in KEEP:
            continue
        if len(parts) >= 5 and parts[2] in ("res", "ck"):
            parts[3] = "run"
        dst = os.path.join(OUT, *parts)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with t.extractfile(m) as src, open(dst, "wb") as f:
            f.write(src.read())
        n += 1
print("extracted", n)
