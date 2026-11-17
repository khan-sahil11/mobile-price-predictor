# UI: choose phone specs -> click Predict -> see price. Loads the saved model only.
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent.parent / "Trained Model" / "model.joblib"
INFO_PATH = Path(__file__).parent.parent / "Trained Model" / "metrics.json"

st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱", layout="centered")
st.markdown("""
<style>
.result-card {background:#FFFFFF; border:1px solid #D9DFE5; border-radius:12px;
              padding:20px; text-align:center;}
.result-price {font-size:2.2rem; font-weight:700; color:#2F3A45; margin:4px 0;}
.result-range {color:#5B7C99; font-size:1rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_info():
    try:
        return json.loads(INFO_PATH.read_text())
    except Exception:
        return {"rmse": 0.0, "brands": ["Apple", "Samsung", "Xiaomi"], "caps": {}}

def inr(x):                                   # 1234567 -> Rs. 12,34,567
    s = str(int(round(x)))
    if len(s) <= 3:
        return "Rs. " + s
    head, tail, parts = s[:-3], s[-3:], []
    while len(head) > 2:
        parts.insert(0, head[-2:]); head = head[:-2]
    if head:
        parts.insert(0, head)
    return "Rs. " + ",".join(parts + [tail])

st.title("📱 Mobile Phone Price Predictor")
st.caption("Choose the specifications and click Predict.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model. Run train_model.py first. Details: {e}")
    st.stop()
info = load_info()

with st.container(border=True):
    c1, c2 = st.columns(2)
    brand = c1.selectbox("Brand", info["brands"], index=None, placeholder="Select a brand")
    ram = c2.selectbox("RAM (GB)", [2, 3, 4, 6, 8, 12, 16], index=3)
    storage = c1.selectbox("Storage (GB)", [16, 32, 64, 128, 256, 512, 1024], index=3)
    screen = c2.slider("Display size (inches)", 4.0, 8.0, 6.5, 0.1)
    battery = c1.slider("Battery (mAh)", 1500, 7000, 5000, 100)
    camera = c2.number_input("Main camera (MP)", min_value=5, max_value=200, value=48, step=1)
    n_cam = c1.selectbox("Number of rear cameras", [1, 2, 3, 4, 5], index=2)
    is_5g = c2.checkbox("5G supported", value=True)
    os_name = "iOS" if brand == "Apple" else "Android"
    st.caption(f"Operating system (auto from brand): **{os_name}**")

if st.button("Predict Price", type="primary"):
    problems = []
    if brand is None:
        problems.append("Please select a brand.")
    if storage < ram:
        problems.append("Storage cannot be smaller than RAM.")
    if problems:
        for p in problems:
            st.error(p)
    else:
        try:
            row = {"RAM_GB": ram, "Storage_GB": storage, "Screen_inch": screen,
                   "Battery_mAh": battery, "Main_Camera_MP": camera,
                   "Num_Cameras": n_cam, "Is_5G": int(is_5g), "Brand": brand, "OS": os_name}
            for col, (lo, hi) in info.get("caps", {}).items():
                row[col] = min(max(row[col], lo), hi)
            pred = max(float(model.predict(pd.DataFrame([row]))[0]), info.get("min_price", 0))
            rmse = float(info.get("rmse", 0))
            st.markdown(f"""
            <div class="result-card">
              <div>Estimated price</div>
              <div class="result-price">{inr(pred)}</div>
              <div class="result-range">Likely range: {inr(max(pred - rmse, 0))} to {inr(pred + rmse)}</div>
            </div>""", unsafe_allow_html=True)
            st.caption("The range is +/- the model's test RMSE. It is an estimate, not an exact price.")
        except Exception as e:
            st.error(f"Prediction failed. Please check your inputs. Details: {e}")