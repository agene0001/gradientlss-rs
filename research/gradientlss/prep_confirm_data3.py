"""Third batch of confirmation datasets (proposal 038): plain numeric CSVs, features first, target last as `y`.

Usage: uv run --no-project --with openpyxl python -I prep_confirm_data3.py <data_confirm dir>

Sources (UCI Machine Learning Repository, fetched 2026-11-14, each unpacked in its own folder `m_<name>`):
  m_energy      242  Energy Efficiency                       y = heating load Y1 (cooling load Y2 dropped)
  m_steel       851  Steel Industry Energy Consumption       y = Usage_kWh; date dropped (NSM, weekday kept); CO2 dropped (computed from
                                                             the target); Load_Type one-hot; 12 000-row subsample
  m_fishtox     504  QSAR fish toxicity                      y = LC50 (-log mol/L)
  m_tempbias    514  Bias correction of temperature forecast y = Next_Tmax; Next_Tmin dropped; Date -> year, day of year; rows with
                                                             missing values dropped
  m_video       335  Online Video Characteristics and Transcoding Time   y = utime (s); id and umem dropped; codec, o_codec one-hot;
                                                             12 000-row subsample
  m_pm25        381  Beijing PM2.5                           y = pm2.5 (integer, count-like); No dropped; cbwd one-hot; NA rows dropped;
                                                             12 000-row subsample
  m_fbcomments  363  Facebook Comment Volume (Variant 1)     y = comments in the next H hours (count); 12 000-row subsample

None of these was used in cycles 1-30. Subsamples use random.Random(2026). The bench reads them as `c_m_<name>`.
"""
import csv
import random
import sys
from datetime import datetime
from pathlib import Path

import openpyxl

root = Path(sys.argv[1])
SUB = 12_000


def write(name, header, rows):
    assert all(len(r) == len(header) for r in rows)
    assert all(v == v and abs(v) != float("inf") for r in rows for v in r)
    with open(root / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    ys = [r[-1] for r in rows]
    ints = all(float(y).is_integer() for y in ys)
    print(f"{name:13s} rows {len(rows):6d}  features {len(header) - 1:3d}  y min {min(ys):.4g} mean {sum(ys) / len(ys):.4g} max {max(ys):.4g}  "
          f"integer y {ints}  zeros {sum(1 for y in ys if y == 0)}")


def sub(rows):
    return rows if len(rows) <= SUB else random.Random(2026).sample(rows, SUB)


# energy efficiency: X1..X8, Y1, Y2
ws = openpyxl.load_workbook(root / "m_energy/ENB2012_data.xlsx", read_only=True, data_only=True).worksheets[0]
it = ws.iter_rows(values_only=True)
h = [c for c in next(it) if c is not None]
assert h[:10] == ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "Y1", "Y2"], h
rows = [[float(v) for v in r[:9]] for r in it if r[0] is not None]
write("m_energy", [f"f{i}" for i in range(8)] + ["y"], rows)

# steel industry
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
with open(root / "m_steel/Steel_industry_data.csv", newline="", encoding="utf-8-sig") as f:
    r = csv.DictReader(f)
    recs = list(r)
loads = sorted({x["Load_Type"] for x in recs})
num = ["Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh", "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor", "NSM"]
rows = [[float(x[c]) for c in num] + [1.0 if x["WeekStatus"] == "Weekend" else 0.0, float(DAYS.index(x["Day_of_week"]))]
        + [1.0 if x["Load_Type"] == c else 0.0 for c in loads] + [float(x["Usage_kWh"])] for x in recs]
rows = [r for r in rows if r[-1] > 0]
write("m_steel", ["lag_kvarh", "lead_kvarh", "lag_pf", "lead_pf", "nsm", "weekend", "weekday"] + [f"load_{c.lower()}" for c in loads] + ["y"], sub(rows))

# QSAR fish toxicity: ';' separated, no header
with open(root / "m_fishtox/qsar_fish_toxicity.csv", newline="") as f:
    rows = [[float(v) for v in row] for row in csv.reader(f, delimiter=";") if row]
write("m_fishtox", [f"f{i}" for i in range(6)] + ["y"], rows)

# temperature forecast bias correction
with open(root / "m_tempbias/Bias_correction_ucl.csv", newline="") as f:
    r = csv.reader(f)
    h = next(r)
    assert h[:2] == ["station", "Date"] and h[-2:] == ["Next_Tmax", "Next_Tmin"]
    rows = []
    for row in r:
        if any(v.strip().lower() in ("", "na", "nan") for v in row):
            continue
        t = datetime.strptime(row[1], "%Y-%m-%d")
        rows.append([float(row[0]), float(t.year), float(t.timetuple().tm_yday)] + [float(v) for v in row[2:-2]] + [float(row[-2])])
write("m_tempbias", ["station", "year", "yday"] + [f"f{i}" for i in range(len(h) - 4)] + ["y"], rows)

# video transcoding time
with open(root / "m_video/transcoding_mesurment.tsv", newline="") as f:
    r = csv.DictReader(f, delimiter="\t")
    recs = list(r)
num = ["duration", "width", "height", "bitrate", "framerate", "i", "p", "b", "frames", "i_size", "p_size", "b_size", "size", "o_bitrate", "o_framerate", "o_width", "o_height"]
cod = sorted({x["codec"] for x in recs} | {x["o_codec"] for x in recs})
rows = [[float(x[c]) for c in num] + [1.0 if x["codec"] == c else 0.0 for c in cod] + [1.0 if x["o_codec"] == c else 0.0 for c in cod] + [float(x["utime"])] for x in recs]
rows = [r for r in rows if r[-1] > 0]
write("m_video", num + [f"in_{c}" for c in cod] + [f"out_{c}" for c in cod] + ["y"], sub(rows))

# Beijing PM2.5
with open(root / "m_pm25/PRSA_data_2010.1.1-2014.12.31.csv", newline="") as f:
    r = csv.DictReader(f)
    recs = [x for x in r if x["pm2.5"] not in ("NA", "")]
wd = sorted({x["cbwd"] for x in recs})
num = ["year", "month", "day", "hour", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
rows = [[float(x[c]) for c in num] + [1.0 if x["cbwd"] == c else 0.0 for c in wd] + [float(x["pm2.5"])] for x in recs]
write("m_pm25", [c.lower() for c in num] + [f"wd_{c.lower()}" for c in wd] + ["y"], sub(rows))

# Facebook comment volume, variant 1: 53 features, target last, no header
with open(root / "m_fbcomments/Dataset/Training/Features_Variant_1.csv", newline="") as f:
    rows = [[float(v) for v in row] for row in csv.reader(f) if row]
assert len(rows[0]) == 54
write("m_fbcomments", [f"f{i}" for i in range(53)] + ["y"], sub(rows))
