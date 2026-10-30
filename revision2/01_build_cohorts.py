"""
Revision 2 - Step 1: build the four TBI cohorts from the RAW source files.

Changes vs. the original preprocessing.ipynb (reviewer #1, comments 1 and 3):
  * Explicit, reproducible TBI eligibility criteria per cohort + cohort-flow counts.
  * NO imputation here. Missing values are left as NaN and imputed later *inside each
    training split only* (02_transfer_experiment.py), and the outcome (LOS) is never
    used as an imputation predictor.
  * Physiologically implausible values are set to missing (and counted).
  * India LOS: the TITCO date-shifted timestamps (years up to ~2290) overflowed the
    pandas nanosecond range in the original run and silently became NaN -> imputed.
    LOS is now computed with day-resolution-safe arithmetic, and patients with a truly
    missing/negative LOS are excluded instead of imputed.

Outputs (revision2/outputs/):
  cohorts/<name>_r2.csv, cohort_flow.csv, missingness.csv, cohort_characteristics.csv
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw_data")
OUT = os.path.join(HERE, "outputs")
os.makedirs(os.path.join(OUT, "cohorts"), exist_ok=True)

# Plausible ranges; values outside are treated as missing (data-entry / not-assessed codes,
# e.g. RR recorded as 0 in intubated patients, GCS recorded as 0 in pre-verbal children).
PLAUSIBLE = {"age": (0, 110), "sbp": (40, 300), "hr": (20, 250), "rr": (4, 80), "gcs": (3, 15)}
COLS = ["age", "sex", "sbp", "hr", "rr", "gcs", "MOI", "hr_cat_given", "rr_cat_given",
        "sbp_cat_given", "los", "died"]

flow, implaus = [], []


def map_moi(text):
    """Unchanged from preprocessing.ipynb (harmonized mechanism-of-injury taxonomy)."""
    if pd.isnull(text):
        return "Other / Unknown"
    text = str(text).strip().lower()
    rta_keywords = ['road traffic injury', 'rta', 'mvc', 'vehicle', 'auto', 'car', 'driver', 'passenger', 'collision']
    if any(kw in text for kw in rta_keywords):
        if 'motorcyclist' in text or 'motorcycle' in text:
            return "Motorcycle Crash"
        if 'bicyclist' in text or 'bike' in text:
            return "Bicycle Crash"
        if 'pedestrian' in text:
            return "Pedestrian Struck"
        return "Motor Vehicle Collision"
    if 'motorcycle' in text or 'motorcyclist' in text or 'motocycle' in text:
        return "Motorcycle Crash"
    if 'bicycle' in text or 'bicyclist' in text or 'bike' in text:
        return "Bicycle Crash"
    if 'pedestrian' in text or 'peds vs auto' in text or 'scooter' in text:
        return "Pedestrian Struck"
    if 'fall' in text or 'fell' in text or 'found down' in text or 'stairs' in text or 'roof' in text:
        return "Fall"
    if 'assault' in text or 'struck' in text or 'stab' in text or 'hit' in text or 'pole' in text or 'collision with horse' in text:
        return "Assault / Blunt Force"
    if 'gsw' in text or 'gunshot' in text:
        return "Gunshot Wound (GSW)"
    if 'fire' in text or 'burn' in text or 'blast' in text or 'explosion' in text:
        return "Burn / Fire / Blast"
    if 'train' in text or 'railway' in text:
        return "Other / Unknown"
    if 'alligator' in text or 'animal' in text or 'boat' in text or 'golf cart' in text or 'atv' in text:
        return "Animal / Environmental"
    return "Other / Unknown"


def step(cohort, label, n):
    flow.append({"cohort": cohort, "step": label, "n": int(n)})


def clean_implausible(df, cohort):
    for col, (lo, hi) in PLAUSIBLE.items():
        if col not in df:
            continue
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        implaus.append({"cohort": cohort, "variable": col, "n_implausible": int(bad.sum()),
                        "rule": f"<{lo} or >{hi}"})
        df.loc[bad, col] = np.nan
    return df


def finish(df, cohort):
    for c in COLS:
        if c not in df:
            df[c] = np.nan
    df = clean_implausible(df[COLS].copy(), cohort)
    df["los"] = df["los"].clip(lower=1)          # strictly positive times (as in original)
    df["discharged"] = (df["died"] == 0).astype(int)
    df.to_csv(os.path.join(OUT, "cohorts", f"{cohort.lower()}_r2.csv"), index=False)
    step(cohort, "Final analytic cohort", len(df))
    return df


# ---------------------------------------------------------------- INDIA (TITCO)
t = pd.read_csv(os.path.join(RAW, "titco-I-full-dataset-v1.csv"), low_memory=False, encoding="latin-1")
step("India", "TITCO registry, all admitted trauma patients", len(t))
icd_cols = [c for c in t.columns if c.endswith("_icd")]
is_tbi = t[icd_cols].astype(str).apply(lambda s: s.str.lower().str.contains("s06")).any(axis=1)
t = t[is_tbi].copy()
step("India", "TBI: >=1 ICD-10 S06 (intracranial injury) code", len(t))
t = t[t["died"].isin(["Yes", "No"])]
step("India", "Excluded: in-hospital vital status missing", len(t))


def to_days(d, tm):
    # TITCO dates are randomly shifted into years beyond the pandas ns range -> parse
    # date parts manually and count days with the proleptic Gregorian ordinal.
    out = []
    for a, b in zip(d, tm):
        try:
            y, m, dd = (int(x) for x in str(a).split("-"))
            hh, mm = (int(x) for x in str(b).split(":")) if isinstance(b, str) and ":" in b else (np.nan, np.nan)
            from datetime import date
            out.append(date(y, m, dd).toordinal() + (hh * 60 + mm) / 1440.0)
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=float)


t["los"] = to_days(t["dodd"], t["todd"]) - to_days(t["doa"], t["toa"])
t = t[t["los"].notna()]
step("India", "Excluded: LOS not computable (missing admission/discharge date-time)", len(t))
t = t[t["los"] >= 0]
step("India", "Excluded: negative LOS (discharge before admission)", len(t))
india = pd.DataFrame({
    "age": pd.to_numeric(t["age"], errors="coerce"),
    "sex": t["sex"].map({"Male": 1, "Female": 0}),
    "sbp": pd.to_numeric(t["sbp_1"], errors="coerce"),
    "hr": pd.to_numeric(t["hr_1"], errors="coerce"),
    "rr": pd.to_numeric(t["rr_1"], errors="coerce"),
    "gcs": pd.to_numeric(t["gcs_t_1"], errors="coerce"),
    "MOI": t["moi"].apply(map_moi),
    "los": t["los"],
    "died": (t["died"] == "Yes").astype(int),
})
india = finish(india, "India")

# ---------------------------------------------------------------- JORDAN (pediatric)
j = pd.read_excel(os.path.join(RAW, "final peds trauma.xlsx"))
step("Jordan", "Pediatric trauma dataset (BIG score), all patients", len(j))
bi = j["any brain injuries"].astype(str).str.strip().str.lower()
j = j[bi.str.startswith("yes")].copy()
step("Jordan", "TBI: 'any brain injuries' = Yes (imaging/clinical diagnosis)", len(j))
out = j["outcome"].astype(str).str.strip().str.lower()
j = j[out.isin(["survival", "died"])]
step("Jordan", "Excluded: in-hospital outcome missing", len(j))
j = j[pd.to_numeric(j["length of stay in the hospital (in days)"], errors="coerce").notna()]
step("Jordan", "Excluded: LOS missing", len(j))


def norm_cat(s, mapping):
    return s.astype(str).str.strip().str.lower().map(mapping)


jordan = pd.DataFrame({
    "age": pd.to_numeric(j["age of diagnosis"], errors="coerce"),
    "sex": j["Gender"].astype(str).str.strip().map({"Male": 1, "Female": 0}),
    "gcs": pd.to_numeric(j["GCS in ER"], errors="coerce"),   # 0 = not assessed -> implausible -> NaN
    "MOI": j["type of trauma"].apply(map_moi),
    # Jordan vitals are only available as categories in the source data
    "hr_cat_given": norm_cat(j["ER-HR"], {"normal": "normal", "noraml": "normal", "tachycardia": "tachycardia"}),
    "rr_cat_given": norm_cat(j["ER-RR"], {"normal": "normal", "noraml": "normal", "tachypnea": "tachypnea"}),
    # harmonized SBP category is low vs not-low; 'high' -> 'normal' (not hypotensive)
    "sbp_cat_given": norm_cat(j["ER-systolic BP"], {"normal": "normal", "high": "normal", "low": "low"}),
    "los": pd.to_numeric(j["length of stay in the hospital (in days)"], errors="coerce"),
    "died": (j["outcome"].astype(str).str.strip().str.lower() == "died").astype(int),
})
jordan = finish(jordan, "Jordan")

# ---------------------------------------------------------------- CALIFORNIA (adult TBI with BAC)
c = pd.read_excel(os.path.join(RAW, "TBI EtOH Submitted Data.xls"), sheet_name="Mortality")
c = c[c["Person"] == 1]   # patient rows only (sheet also holds summary rows)
step("California", "Adult TBI admissions with blood alcohol tested (published dataset)", len(c))
c = c[c["LOS"].notna() & c["Mortality"].notna()]
step("California", "Excluded: LOS or in-hospital outcome missing", len(c))
california = pd.DataFrame({
    "age": c["Age"], "sex": c["Sex"].map({"M": 1, "F": 0}),
    "sbp": c["SBP"], "hr": c["HR"], "rr": c["RR"], "gcs": c["Trauma GCS"],
    "MOI": c["Cause of Injury"].apply(map_moi),
    "los": c["LOS"], "died": c["Mortality"].astype(int),
})
california = finish(california, "California")

# ---------------------------------------------------------------- FLORIDA (institutional; not redistributed)
# Raw institutional extract is not in this repository. florida_clean.csv equals the raw
# extract except for the IterativeImputer-filled cells, which are identifiable as
# non-integer values (all raw vitals/GCS/age are integers) and are reset to missing here.
fl_path = os.path.join(HERE, "..", "florida_clean.csv")
if os.path.exists(fl_path):
    f = pd.read_csv(fl_path)
    step("Florida", "Adults (>=18 y) with blunt TBI admitted to ICU (institutional registry)", len(f))
    for col in ["age", "sbp", "hr", "rr", "gcs"]:
        f.loc[f[col] % 1 != 0, col] = np.nan
    florida = pd.DataFrame({k: f[k] for k in ["age", "sex", "sbp", "hr", "rr", "gcs", "MOI", "los"]})
    florida["died"] = f["event"].astype(int)
    florida = finish(florida, "Florida")
else:
    print("florida_clean.csv not found - Florida cohort skipped")

# ---------------------------------------------------------------- reports
pd.DataFrame(flow).to_csv(os.path.join(OUT, "cohort_flow.csv"), index=False)

rows = []
for name in ["India", "Jordan", "Florida", "California"]:
    p = os.path.join(OUT, "cohorts", f"{name.lower()}_r2.csv")
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p)
    n = len(df)
    for var in ["age", "sex", "sbp", "hr", "rr", "gcs", "MOI", "los", "died"]:
        if name == "Jordan" and var in ["sbp", "hr", "rr"]:
            miss = int(df[f"{var}_cat_given"].isna().sum())
            label = f"{var} (category only)"
        else:
            miss = int(df[var].isna().sum())
            label = var
        imp = [r["n_implausible"] for r in implaus if r["cohort"] == name and r["variable"] == var]
        rows.append({"cohort": name, "variable": label, "N": n,
                     "implausible_set_missing": imp[0] if imp else 0,
                     "missing_total_n": miss, "missing_total_pct": round(100 * miss / n, 1)})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "missingness.csv"), index=False)

# characteristics: LOS as median (IQR) - reviewer minor comment
char = []
for name in ["India", "Jordan", "Florida", "California"]:
    p = os.path.join(OUT, "cohorts", f"{name.lower()}_r2.csv")
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p)
    q = df["los"].quantile([.25, .5, .75])
    qd = df.loc[df.discharged == 1, "los"].quantile([.25, .5, .75])
    a = df["age"].quantile([.25, .5, .75])
    char.append({
        "cohort": name, "N": len(df),
        "age_median_IQR": f"{a[.5]:.0f} ({a[.25]:.0f}-{a[.75]:.0f})" if name != "Jordan" else f"{a[.5]:.1f} ({a[.25]:.1f}-{a[.75]:.1f})",
        "male_pct": round(100 * df.sex.mean(), 1),
        "gcs_median_IQR": "{:.0f} ({:.0f}-{:.0f})".format(*df.gcs.quantile([.5, .25, .75])),
        "LOS_all_median_IQR": f"{q[.5]:.1f} ({q[.25]:.1f}-{q[.75]:.1f})",
        "LOS_discharged_median_IQR": f"{qd[.5]:.1f} ({qd[.25]:.1f}-{qd[.75]:.1f})",
        "LOS_mean_SD": f"{df.los.mean():.1f} ± {df.los.std():.1f}",
        "discharged_alive_n": int(df.discharged.sum()),
        "in_hospital_deaths_n": int(df.died.sum()),
        "mortality_pct": round(100 * df.died.mean(), 1),
    })
pd.DataFrame(char).to_csv(os.path.join(OUT, "cohort_characteristics.csv"), index=False)

print(pd.DataFrame(flow).to_string(index=False))
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(char).T.to_string())
