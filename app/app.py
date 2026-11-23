# Used Vehicle Price Estimator - Streamlit application (CDS4003 Data Science Final Project)
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import vehicle_features as vf          # the same feature code that was used during training

st.set_page_config(page_title="Used Vehicle Price Estimator", layout="wide")


@st.cache_resource
def load_artifacts():
    price_model = joblib.load(ROOT / "models" / "price_model.joblib")
    meta = json.loads((ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    return price_model, meta


try:
    price_model, meta = load_artifacts()
except Exception as error:
    st.error(f"The model could not be loaded: {error}")
    st.stop()

cats = meta["categories"]


def options(name):
    return cats.get(name) or ["unknown"]


def choose(label, name):
    # Show a menu only when the model knows more than one category (a field that is empty in the dataset is hidden).
    opts = options(name)
    return opts[0] if len(opts) == 1 else st.selectbox(label, opts)


st.title("Used Vehicle Price Estimator")
st.write("Estimate the typical **asking price** of a used car or truck in the US market, with an indicative price range, "
         "and check whether a seller's asking price looks fair.")

tab_estimate, tab_model, tab_business = st.tabs(["Estimate a price", "About the model", "Business use"])

with tab_estimate:
    left, right = st.columns(2)
    with left:
        manufacturer = st.selectbox("Manufacturer", options("manufacturer"))
        model_options = meta["models_by_manufacturer"].get(manufacturer, []) + ["unknown"]
        model_name = st.selectbox("Model", model_options)
        year = st.number_input("Model year", min_value=1950, max_value=vf.REFERENCE_YEAR + 1,
                               value=max(1950, vf.REFERENCE_YEAR - 6), step=1)
        odometer = st.number_input("Odometer (miles)", min_value=0, max_value=1_000_000, value=70_000, step=1_000)
        condition = choose("Condition", "condition")
        title_status = choose("Title status", "title_status")
        state = choose("State", "state")
    with right:
        vehicle_type = choose("Vehicle type", "type")
        fuel = choose("Fuel", "fuel")
        transmission = choose("Transmission", "transmission")
        drive = choose("Drive", "drive")
        cylinders = choose("Cylinders", "cylinders")
        size = choose("Size", "size")
        paint_color = choose("Paint colour", "paint_color")
        seller_type = choose("Seller type", "seller_type")

    asking_price = st.number_input("Asking price to check (optional, US dollars)", min_value=0, value=0, step=500)

    if st.button("Estimate price", type="primary"):
        values = {"year": year, "odometer": odometer, "manufacturer": manufacturer, "model": model_name,
                  "condition": condition, "cylinders": cylinders, "fuel": fuel, "title_status": title_status,
                  "transmission": transmission, "drive": drive, "size": size, "type": vehicle_type,
                  "paint_color": paint_color, "seller_type": seller_type, "state": state}
        problems = vf.validate_inputs(values)
        if problems:
            for message in problems:
                st.error(message)
        else:
            result = vf.predict_price(price_model, values, meta["ratio_low"], meta["ratio_high"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Estimated asking price", f"${result['estimate']:,.0f}")
            c2.metric("Likely range (low)", f"${result['low']:,.0f}")
            c3.metric("Likely range (high)", f"${result['high']:,.0f}")
            st.caption("The range covers about 80% of historical asking prices for similar estimates.")

            age = max(vf.REFERENCE_YEAR - year, 1)
            if odometer / age > 40_000:
                st.warning("This mileage is unusually high for the vehicle's age; the estimate may be less reliable.")
            if manufacturer == "unknown" or model_name == "unknown":
                st.warning("Manufacturer or model is unknown, so the estimate is less specific.")

            if asking_price > 0:
                difference = (asking_price - result["estimate"]) / result["estimate"] * 100
                if asking_price < result["low"]:
                    st.success(f"The asking price is {abs(difference):.0f}% below the estimate - possibly a bargain, "
                               "or the vehicle has a problem that the inputs do not capture.")
                elif asking_price > result["high"]:
                    st.error(f"The asking price is {difference:.0f}% above the estimate and outside the likely range - "
                             "consider negotiating or comparing other listings.")
                else:
                    st.info(f"The asking price is within the likely range ({difference:+.0f}% from the estimate).")

with tab_model:
    st.subheader("How the estimate is made")
    st.write(f"A **{meta['model_name']}** model was trained on {meta['training_rows']:,} cleaned Craigslist listings "
             f"(data collected around {meta['reference_year']}). It predicts the logarithm of the price, so errors are "
             "judged in percentage terms.")
    metrics = meta["test_metrics"]
    table = pd.DataFrame({"Metric": ["R-squared", "Median absolute percentage error (%)", "Predictions within 20% (%)",
                                     "Average absolute error ($)"],
                          "Test-set value": [metrics["R2"], metrics["MdAPE (%)"], metrics["Within 20% (%)"], metrics["MAE ($)"]]})
    st.dataframe(table, hide_index=True)
    image_path = ROOT / "docs" / "feature_importance.png"
    if image_path.exists():
        st.image(str(image_path), caption="Which inputs the model relies on most (permutation importance)")
    st.subheader("Limitations")
    st.write("- The data contains **asking prices**, not final sale prices.\n"
             "- Listings come from one website and one time period; market prices change over time.\n"
             "- Seller-entered data can contain errors; rare and luxury vehicles are estimated less accurately.\n"
             "- The estimate does not know about accidents, modifications or the exact vehicle condition.")

with tab_business:
    st.subheader("How the tool supports decisions")
    st.write("- **Sellers:** choose a competitive asking price faster and avoid pricing far outside the market.\n"
             "- **Buyers:** check whether an asking price is fair before negotiating.\n"
             "- **Dealerships and marketplaces:** price inventory consistently and flag listings priced far from the typical range for review.")
    st.write("Use the estimate as a **reference point**, not as a guaranteed selling price, and re-train the model regularly "
             "with fresh listings because used-car prices drift.")
