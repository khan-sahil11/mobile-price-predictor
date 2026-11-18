# prepare_data.py
# Purpose: read the raw dataset, pick the useful columns, give them the names the
# model expects, fix small problems, and save clean_phones.csv for training.

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent                     # project folder, works on any laptop
raw_path = BASE / "Dataset" / "smartphone_cleaned_v5.csv"
out_path = BASE / "Dataset" / "clean_phones.csv"

df = pd.read_csv(raw_path)
print("Raw shape:", df.shape)

# Brands are lowercase in the raw file ("oneplus"). Map special ones to proper
# display names so the app dropdown looks right. Others just get Title Case.
# "redmi" is Xiaomi's own sub-brand, so we count it as Xiaomi.
SPECIAL = {"oneplus": "OnePlus", "iqoo": "iQOO", "poco": "POCO", "oppo": "OPPO",
           "tecno": "Tecno", "itel": "itel", "ikall": "iKall", "lg": "LG",
           "zte": "ZTE", "tcl": "TCL", "lyf": "Lyf", "blu": "BLU", "cat": "CAT",
           "redmi": "Xiaomi"}
brand = df["brand_name"].astype(str).str.strip().str.lower()

# Build the clean table: keep only the useful columns and rename them
clean = pd.DataFrame()
clean["Brand"] = brand.map(SPECIAL).fillna(brand.str.title())
clean["RAM_GB"] = df["ram_capacity"]
clean["Storage_GB"] = df["internal_memory"]
clean["Screen_inch"] = df["screen_size"]
clean["Battery_mAh"] = df["battery_capacity"]
clean["Main_Camera_MP"] = df["primary_camera_rear"]
clean["Num_Cameras"] = df["num_rear_cameras"]
clean["OS"] = np.where(clean["Brand"] == "Apple", "iOS", "Android")   # Apple = iOS, else Android
clean["Is_5G"] = df["has_5g"].astype(int)                             # True/False -> 1/0
clean["Price_INR"] = df["price"]                                      # target, already in rupees

# Keep only mainstream brands that are relevant today.
# Old or discontinued brands (Blackberry, iKall, Letv, Vertu etc.) are removed so the
# app dropdown is short and the model learns from brands people actually buy.
KEEP = ["Apple", "Samsung", "Xiaomi", "POCO", "Realme", "OnePlus", "OPPO", "Vivo",
        "iQOO", "Motorola", "Google", "Nothing", "Tecno", "Infinix", "Nokia",
        "Lava", "Asus", "itel"]
before_brand = len(clean)
clean = clean[clean["Brand"].isin(KEEP)]
print("Removed rows from minor/old brands:", before_brand - len(clean))

# Missing battery values (a few rows, mostly iPhones, which don't list mAh):
# fill with the median battery of the SAME brand, so an iPhone gets an iPhone-like value
clean["Battery_mAh"] = clean.groupby("Brand")["Battery_mAh"].transform(
    lambda s: s.fillna(s.median()))
clean["Battery_mAh"] = clean["Battery_mAh"].fillna(clean["Battery_mAh"].median())  # safety net

# Remove luxury/collector phones above Rs. 2,00,000 (e.g. Vertu at 6,50,000).
# They are not normal phones and would distort the model.
before = len(clean)
clean = clean[clean["Price_INR"] <= 200000]
print("Removed luxury outliers:", before - len(clean))

clean = clean.drop_duplicates()                       # remove repeated rows

print("Clean shape:", clean.shape)
print("Brands kept:", clean["Brand"].nunique())
print(clean.dtypes)
print(clean.isnull().sum())
print(clean.head())
clean.to_csv(out_path, index=False)
print("Saved to:", out_path)