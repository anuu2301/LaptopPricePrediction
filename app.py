
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "laptop_model.joblib"

st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="centered"
)

st.title("💻 Laptop Price Prediction")
st.write(
    "Enter your laptop specifications to estimate its price "
    "using a Machine Learning model."
)

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    saved = load_model()
    model = saved["model"]
    metadata = saved["metadata"]
except FileNotFoundError:
    st.error(
        "Trained model not found. Run python train_model.py first."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

features = metadata["features"]
numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]
ranges = metadata["ranges"]
categories = metadata["categories"]

st.subheader("Laptop Specifications")

# Collect user inputs
with st.form("prediction_form"):
    user_inputs = {}

    for col in features:
        if col in numeric_features:
            details = ranges[col]
            minimum = details["min"]
            maximum = details["max"]
            default = details["median"]

            # Give number inputs a valid range
            if minimum == maximum:
                maximum = minimum + 1

            step = 1.0 if col == "Ram" else 0.1

            user_inputs[col] = st.number_input(
                col,
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(max(minimum, min(default, maximum))),
                step=step
            )

        else:
            options = categories.get(col, [])

            if options:
                user_inputs[col] = st.selectbox(
                    col,
                    options=options
                )
            else:
                user_inputs[col] = st.text_input(col)

    submitted = st.form_submit_button(
        "Predict Laptop Price",
        use_container_width=True
    )

# Predict the price
if submitted:
    input_data = pd.DataFrame([user_inputs], columns=features)

    try:
        predicted_price = float(model.predict(input_data)[0])

        if predicted_price <= 0:
            st.warning(
                "The model predicted a non-positive price. "
                "Try different specifications or review the dataset."
            )
        else:
            st.success("Prediction completed!")
            st.metric(
                "Estimated Laptop Price",
                f"{predicted_price:,.2f}"
            )
            st.caption(
                "The displayed value uses the same price units "
                "as the training dataset. This is an estimate, "
                "not a guaranteed market price."
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# Show evaluation details
with st.expander("About the trained model"):
    st.write("Algorithm: Multiple Linear Regression")
    st.write(f"Target column: {metadata['target']}")
    st.write(f"Features used: {', '.join(features)}")
    st.write(f"Test-set MAE: {metadata['mae']:.2f}")
    st.write(f"Test-set R²: {metadata['r2']:.3f}")
