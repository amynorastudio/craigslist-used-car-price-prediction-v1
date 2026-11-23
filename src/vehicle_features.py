
# vehicle_features.py - shared feature engineering for training (notebook) and deployment (Streamlit app).
import numpy as np
import pandas as pd

REFERENCE_YEAR = 2026      # data-collection year (fixed constant, NOT learned from prices)
PRICE_CAP = 1000000                # plausibility cap used in Stage 3

LUXURY_BRANDS = {"acura", "alfa-romeo", "aston-martin", "audi", "bentley", "bmw", "cadillac", "ferrari", "infiniti",
                 "jaguar", "land rover", "lexus", "lincoln", "maserati", "mercedes-benz", "porsche", "rolls-royce",
                 "tesla", "volvo"}

_REGIONS = {
    "northeast": "ct me ma nh ri vt nj ny pa",
    "midwest": "il in mi oh wi ia ks mn mo ne nd sd",
    "south": "de fl ga md nc sc va dc wv al ky ms tn ar la ok tx",
    "west": "az co id mt nv nm ut wy ak ca hi or wa",
}
STATE_REGION = {state: region for region, states in _REGIONS.items() for state in states.split()}

RAW_INPUT_COLUMNS = ["year", "odometer", "manufacturer", "model", "condition", "cylinders", "fuel",
                     "title_status", "transmission", "drive", "size", "type", "paint_color", "seller_type", "state"]
CATEGORICAL_RAW = ["manufacturer", "model", "condition", "cylinders", "fuel", "title_status",
                   "transmission", "drive", "size", "type", "paint_color", "seller_type", "state"]
NUMERIC_FEATURES = ["year", "odometer", "vehicle_age", "miles_per_year", "cylinders_num",
                    "is_luxury_brand", "is_electrified", "title_issue"]
CATEGORICAL_FEATURES = CATEGORICAL_RAW + ["state_region"]


def add_features(X):
    # Return a copy of X with cleaned categories and the engineered features added.
    X = X.copy()
    for col in ("year", "odometer"):
        if col not in X.columns:
            X[col] = np.nan
    year = pd.to_numeric(X["year"], errors="coerce").astype("float64")
    odometer = pd.to_numeric(X["odometer"], errors="coerce").astype("float64")
    X["year"], X["odometer"] = year, odometer

    for col in CATEGORICAL_RAW:
        if col not in X.columns:                 # column absent in this dataset version -> treated as unknown
            X[col] = np.nan
        values = X[col].astype("object")
        values = values.where(values.notna(), "unknown")
        X[col] = values.astype(str).str.strip().str.lower()

    age = (REFERENCE_YEAR - year).clip(lower=0)
    X["vehicle_age"] = age
    X["miles_per_year"] = odometer / age.clip(lower=1)
    X["cylinders_num"] = pd.to_numeric(X["cylinders"].str.extract(r"(\d+)")[0], errors="coerce")
    X["is_luxury_brand"] = X["manufacturer"].isin(LUXURY_BRANDS).astype(int)
    X["is_electrified"] = X["fuel"].isin(["electric", "hybrid"]).astype(int)
    X["title_issue"] = (~X["title_status"].isin(["clean", "unknown"])).astype(int)
    X["state_region"] = X["state"].map(STATE_REGION).fillna("unknown")
    return X


def inverse_log_price(values):
    # Convert log(1 + price) predictions back to dollars; clipping avoids absurd outputs for extreme inputs.
    return np.expm1(np.clip(values, 0.0, np.log1p(PRICE_CAP)))


def build_input_frame(values):
    # One-row DataFrame from a dictionary of user inputs.
    return pd.DataFrame([{c: values.get(c, np.nan) for c in RAW_INPUT_COLUMNS}])


def validate_inputs(values):
    # Return a list of problems with the user's inputs (empty list = valid).
    problems = []
    year, odometer = values.get("year"), values.get("odometer")
    if year is None or not (1950 <= year <= REFERENCE_YEAR + 1):
        problems.append(f"Model year must be between 1950 and {REFERENCE_YEAR + 1}.")
    if odometer is None or not (0 <= odometer <= 1_000_000):
        problems.append("Odometer must be between 0 and 1,000,000 miles.")
    if not str(values.get("manufacturer", "")).strip():
        problems.append("Please choose a manufacturer.")
    return problems


def predict_price(model, values, ratio_low=None, ratio_high=None):
    # Estimate the asking price (US dollars) and an optional range from empirical error ratios.
    estimate = float(model.predict(build_input_frame(values))[0])
    result = {"estimate": estimate}
    if ratio_low is not None and ratio_high is not None:
        result["low"], result["high"] = estimate * ratio_low, estimate * ratio_high
    return result
