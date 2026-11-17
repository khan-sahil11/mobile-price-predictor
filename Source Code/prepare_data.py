# prepare_data.py
# Purpose: read the raw CSV, convert text columns to numbers using regex,
# and save a clean CSV that is ready for model training.

import re                      # regex library: finds patterns inside text
import numpy as np
import pandas as pd
from pathlib import Path

USD_TO_INR = 85                # assumed exchange rate; change it if you want

# Build file paths relative to this file (works on any laptop)
BASE = Path(__file__).parent.parent
raw_path = BASE / "Dataset" / "Mobile phone price.csv"
out_path = BASE / "Dataset" / "clean_phones.csv"

# utf-8-sig removes the invisible BOM character stuck to the first column name
df = pd.read_csv(raw_path, encoding="utf-8-sig")

# Some headers have trailing spaces ("Storage ", "RAM "), so strip them
df.columns = df.columns.str.strip()
print("Columns:", df.columns.tolist())
print("Rows before cleaning:", len(df))


# ---------- regex helper functions ----------

def first_number(text):
    """Return the first number in a text, e.g. '6.67 inches' -> 6.67"""
    text = str(text).replace(",", "")                # '1,199' -> '1199'
    match = re.search(r"\d+(?:\.\d+)?", text)        # digits, optional .digits
    return float(match.group()) if match else np.nan

def to_gb(text):
    """'128 GB' -> 128, '1 TB' -> 1024, '512 MB' -> 0.5"""
    text = str(text).upper().replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(TB|GB|MB)?", text)
    if not match:
        return np.nan
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "TB":
        value *= 1024                                # 1 TB = 1024 GB
    elif unit == "MB":
        value /= 1024
    return value

def camera_info(text):
    """'108 + 10 + 10 + 12' -> (main=108, count=4)"""
    numbers = re.findall(r"\d+(?:\.\d+)?", str(text))   # all numbers in the text
    if not numbers:
        return pd.Series([np.nan, np.nan])
    return pd.Series([float(numbers[0]), len(numbers)])


# ---------- build the clean table ----------
clean = pd.DataFrame()
clean["Brand"] = df["Brand"].astype(str).str.strip()

clean["RAM_GB"] = df["RAM"].apply(to_gb)
clean["Storage_GB"] = df["Storage"].apply(to_gb)
clean["Screen_inch"] = df["Screen Size (inches)"].apply(first_number)
clean["Battery_mAh"] = pd.to_numeric(df["Battery Capacity (mAh)"], errors="coerce")
clean[["Main_Camera_MP", "Num_Cameras"]] = df["Camera (MP)"].apply(camera_info)

# New feature 1: OS from brand (Apple -> iOS, others -> Android)
clean["OS"] = np.where(clean["Brand"].str.lower() == "apple", "iOS", "Android")

# New feature 2: is the phone 5G? (look for '5G' in the model name)
clean["Is_5G"] = df["Model"].str.contains(r"5G", case=False, regex=True).astype(int)

# Target: price in USD -> number -> INR
clean["Price_INR"] = df["Price ($)"].apply(first_number) * USD_TO_INR

# ---------- final cleaning ----------
clean = clean.dropna(subset=["Price_INR"])           # a row without target is useless
clean = clean.drop_duplicates()                      # remove repeated rows

print("Rows after cleaning:", len(clean))
print(clean.dtypes)
print(clean.isnull().sum())
print(clean.head())

clean.to_csv(out_path, index=False)
print("Saved to:", out_path)