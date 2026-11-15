"""Second batch of confirmation datasets (proposal 032): plain numeric CSVs, features first, target last as `y`.

Usage: uv run --no-project --with xlrd python -I prep_confirm_data2.py <data_confirm dir>

Sources (UCI Machine Learning Repository, fetched 2026-11-14, each unpacked in its own folder `n_<name>`):
  n_concrete    165  Concrete Compressive Strength       y = strength (MPa)
  n_wine        186  Wine Quality (white)                y = quality score (3-9)
  n_appliances  374  Appliances Energy Prediction        y = Appliances (Wh); date -> hour, weekday; rv1, rv2 dropped; 12 000-row subsample
  n_sgemm       440  SGEMM GPU kernel performance        y = mean of the four run times (ms); 12 000-row subsample
  n_gridstab    471  Electrical Grid Stability Simulated y = stab (the class label stabf dropped)
  n_metro       492  Metro Interstate Traffic Volume     y = traffic_volume (count); weather_main one-hot, holiday flag,
                                                         date_time -> hour, weekday, month, year; 12 000-row subsample
  n_news        332  Online News Popularity              y = shares (count); url and timedelta dropped; 12 000-row subsample

None of these was used in cycles 1-26. Subsamples use random.Random(2026). The bench reads them as `c_n_<name>`.
"""
import csv
import random
import sys
from datetime import datetime
from pathlib import Path

import xlrd

root = Path(sys.argv[1])
SUB = 12_000


def write(name, header, rows):
    assert all(len(r) == len(header) for r in rows)
    with open(root / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    ys = [r[-1] for r in rows]
    print(f"{name:13s} rows {len(rows):6d}  features {len(header) - 1:3d}  y min {min(ys):.4g} mean {sum(ys) / len(ys):.4g} max {max(ys):.4g}")


def sub(rows):
    return rows if len(rows) <= SUB else random.Random(2026).sample(rows, SUB)


# concrete: one sheet, 8 features then the strength
sh = xlrd.open_workbook(str(root / "n_concrete/Concrete_Data.xls")).sheet_by_index(0)
rows = [[float(v) for v in sh.row_values(i)] for i in range(1, sh.nrows)]
write("n_concrete", [f"f{i}" for i in range(sh.ncols - 1)] + ["y"], rows)

# white wine: ';' separated
with open(root / "n_wine/winequality-white.csv", newline="") as f:
    r = csv.reader(f, delimiter=";")
    h = next(r)
    rows = [[float(v) for v in row] for row in r if row]
assert h[-1] == "quality"
write("n_wine", [f"f{i}" for i in range(len(h) - 1)] + ["y"], rows)

# appliances energy
with open(root / "n_appliances/energydata_complete.csv", newline="") as f:
    r = csv.reader(f)
    h = next(r)
    keep = [i for i, c in enumerate(h) if c not in ("date", "Appliances", "rv1", "rv2")]
    yi = h.index("Appliances")
    rows = []
    for row in r:
        t = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        rows.append([float(row[i]) for i in keep] + [t.hour + t.minute / 60.0, float(t.weekday()), float(row[yi])])
write("n_appliances", [f"f{i}" for i in range(len(keep))] + ["hour", "weekday", "y"], sub(rows))

# SGEMM: 14 parameters, four timings
with open(root / "n_sgemm/sgemm_product.csv", newline="") as f:
    r = csv.reader(f)
    h = next(r)
    rows = [[float(v) for v in row[:14]] + [sum(float(v) for v in row[14:18]) / 4.0] for row in r]
assert h[14].startswith("Run1")
write("n_sgemm", [f"f{i}" for i in range(14)] + ["y"], sub(rows))

# grid stability
with open(root / "n_gridstab/Data_for_UCI_named.csv", newline="") as f:
    r = csv.reader(f)
    h = next(r)
    assert h[-2:] == ["stab", "stabf"]
    rows = [[float(v) for v in row[:-1]] for row in r]
write("n_gridstab", [f"f{i}" for i in range(len(h) - 2)] + ["y"], rows)

# metro traffic volume
with open(root / "n_metro/Metro_Interstate_Traffic_Volume.csv", newline="") as f:
    r = csv.reader(f)
    h = next(r)
    recs = list(r)
ix = {c: i for i, c in enumerate(h)}
cats = sorted({row[ix["weather_main"]] for row in recs})
rows = []
for row in recs:
    t = datetime.strptime(row[ix["date_time"]], "%Y-%m-%d %H:%M:%S")
    x = [0.0 if row[ix["holiday"]] in ("None", "") else 1.0] + [float(row[ix[c]]) for c in ("temp", "rain_1h", "snow_1h", "clouds_all")]
    x += [1.0 if row[ix["weather_main"]] == c else 0.0 for c in cats]
    x += [float(t.hour), float(t.weekday()), float(t.month), float(t.year)]
    rows.append(x + [float(row[ix["traffic_volume"]])])
write("n_metro", ["holiday", "temp", "rain_1h", "snow_1h", "clouds_all"] + [f"w_{c.lower()}" for c in cats] + ["hour", "weekday", "month", "year", "y"], sub(rows))

# online news popularity: header fields carry a leading space
with open(root / "n_news/OnlineNewsPopularity/OnlineNewsPopularity.csv", newline="") as f:
    r = csv.reader(f)
    h = [c.strip() for c in next(r)]
    assert h[0] == "url" and h[1] == "timedelta" and h[-1] == "shares"
    rows = [[float(v) for v in row[2:]] for row in r if row]
write("n_news", [f"f{i}" for i in range(len(h) - 3)] + ["y"], sub(rows))
