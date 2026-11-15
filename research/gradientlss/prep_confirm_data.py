"""Turn the downloaded confirmation datasets into plain numeric CSVs: features first, target last as `y`.

Usage: uv run --no-project python -I prep_confirm_data.py <data_confirm dir>

Sources (UCI Machine Learning Repository, fetched 2026-11-13, each unpacked in its own folder):
  airfoil     291  Airfoil Self-Noise                  y = sound pressure level (dB)
  supercond   464  Superconductivty Data               y = critical temperature (K); 12 000-row subsample
  parkinsons  189  Parkinsons Telemonitoring           y = total_UPDRS (motor_UPDRS and subject id dropped)
  gasturbine  551  Gas Turbine CO and NOx Emission     y = NOx (gt_nox) or CO (gt_co); 12 000-row subsample
  abalone       1  Abalone                             y = rings (count); sex one-hot
  seoulbike   560  Seoul Bike Sharing Demand           y = rented bikes per hour (count); non-functioning days dropped

None of these was used in cycles 1-13. Subsamples use random.Random(2026).
"""
import csv
import random
import sys
from datetime import date
from pathlib import Path

root = Path(sys.argv[1])
SUB = 12_000


def write(name, header, rows):
    with open(root / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    ys = [r[-1] for r in rows]
    print(f"{name:11s} rows {len(rows):6d}  features {len(header) - 1:3d}  y min {min(ys):.4g} mean {sum(ys) / len(ys):.4g} max {max(ys):.4g}")


def sub(rows):
    if len(rows) <= SUB:
        return rows
    return random.Random(2026).sample(rows, SUB)


# airfoil: tab separated, no header
rows = [[float(v) for v in line.split()] for line in open(root / "airfoil/airfoil_self_noise.dat") if line.strip()]
write("airfoil", ["freq", "angle", "chord", "velocity", "thickness", "y"], rows)

# superconductivity
with open(root / "supercond/train.csv", newline="") as f:
    r = csv.reader(f)
    h = next(r)
    rows = [[float(v) for v in row] for row in r]
assert h[-1] == "critical_temp"
write("supercond", [f"f{i}" for i in range(len(h) - 1)] + ["y"], sub(rows))

# parkinsons telemonitoring
with open(root / "parkinsons/parkinsons_updrs.data", newline="") as f:
    r = csv.reader(f)
    h = next(r)
    keep = [i for i, c in enumerate(h) if c not in ("subject#", "motor_UPDRS", "total_UPDRS")]
    yi = h.index("total_UPDRS")
    rows = [[float(row[i]) for i in keep] + [float(row[yi])] for row in r]
write("parkinsons", [f"f{i}" for i in range(len(keep))] + ["y"], rows)

# gas turbine: five yearly files
rows_nox, rows_co = [], []
for yr in range(2011, 2016):
    with open(root / f"gasturbine/gt_{yr}.csv", newline="") as f:
        r = csv.reader(f)
        h = next(r)
        fi = [i for i, c in enumerate(h) if c not in ("CO", "NOX")]
        for row in r:
            x = [float(row[i]) for i in fi]
            rows_nox.append(x + [float(row[h.index("NOX")])])
            rows_co.append(x + [float(row[h.index("CO")])])
hdr = [h[i] for i in fi] + ["y"]
write("gt_nox", hdr, sub(rows_nox))
write("gt_co", hdr, sub(rows_co))

# abalone: sex one-hot, rings as the count target
rows = []
for line in open(root / "abalone/abalone.data"):
    p = line.strip().split(",")
    if len(p) != 9:
        continue
    rows.append([float(p[0] == s) for s in "MFI"] + [float(v) for v in p[1:8]] + [float(p[8])])
write("abalone", ["sex_m", "sex_f", "sex_i", "length", "diameter", "height", "whole", "shucked", "viscera", "shell", "y"], rows)

# Seoul bike: hourly counts; drop the non-functioning days (count is 0 by construction there)
rows = []
with open(root / "seoulbike/SeoulBikeData.csv", newline="", encoding="latin-1") as f:
    r = csv.reader(f)
    next(r)
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    for row in r:
        if row[13].strip() != "Yes":
            continue
        d, m, y = (int(v) for v in row[0].split("/"))
        dt = date(y, m, d)
        x = [float(row[2])] + [float(v) for v in row[3:11]]
        x += [float(seasons.index(row[11])), float(row[12].strip() == "Holiday"), float(m), float(dt.weekday())]
        rows.append(x + [float(row[1])])
write("seoulbike", ["hour", "temp", "humidity", "wind", "visibility", "dewpoint", "solar", "rain", "snow", "season", "holiday",
                    "month", "weekday", "y"], rows)
