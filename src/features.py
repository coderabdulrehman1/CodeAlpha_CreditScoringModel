import numpy as np, pandas as pd

LATE_COLS = ["late_30_59", "late_60_89", "late_90"]
UTIL_CAP =  1.092956 
DEBT_CAP =  2449.000000  

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["placeholder_code"] = df[LATE_COLS].isin([96, 98]).any(axis=1).astype(int)
    df[LATE_COLS] = df[LATE_COLS].replace([96, 98], np.nan)
    df["income_missing"] = df["income"].isna().astype(int)
    df["total_late"] = df[LATE_COLS].sum(axis=1)
    df["any_90_late"] = (df["late_90"] > 0).astype(int)
    df["monthly_debt"] = df["debt_ratio"] * df["income"]
    df["income_per_person"] = df["income"] / (df["dependents"] + 1)
    df["over_limit"] = (df["utilization"] > 1).astype(int)
    df["utilization"] = df["utilization"].clip(upper=UTIL_CAP)
    df["debt_ratio"] = df["debt_ratio"].clip(upper=DEBT_CAP)
    df["log_income"] = np.log1p(df["income"])
    return df