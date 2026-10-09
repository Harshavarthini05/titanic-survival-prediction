import numpy as np
import streamlit as st
import shap
import matplotlib.pyplot as plt

from titanic_inference import (
    load_model_and_features,
    preprocess_one_passenger,
)


# Page configuration must come before other Streamlit commands.
st.set_page_config(
    page_title="Titanic AI | Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Professional styling. This CSS does not hide the sidebar.
st.markdown(
    """
    <style>
        .stApp {
            background-color: #F4F8FC;
            color: #14213D;
        }

        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Sidebar appearance only; it remains visible. */
        [data-testid="stSidebar"] {
            background: #0B2D50;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: #FFFFFF !important;
        }

        /* Keep sidebar input backgrounds white and their text dark. */
        [data-testid="stSidebar"] input {
            background-color: #FFFFFF !important;
            color: #14213D !important;
            -webkit-text-fill-color: #14213D !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] * {
            color: #14213D !important;
        }

        .hero-card {
            background: linear-gradient(135deg, #071A33, #1479A8);
            padding: 2.2rem 2.5rem;
            border-radius: 18px;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 24px rgba(7, 26, 51, 0.16);
        }

        .hero-badge {
            display: inline-block;
            color: #DCEEFF;
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 30px;
            padding: 0.35rem 0.75rem;
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
        }

        .hero-title {
            color: #FFFFFF !important;
            font-size: 2.35rem;
            font-weight: 800;
            margin: 0;
        }

        .hero-subtitle {
            color: #E5F3FF !important;
            font-size: 1.05rem;
            line-height: 1.6;
            margin-top: 0.65rem;
        }

        .info-card {
            background: #FFFFFF;
            border: 1px solid #DFE8F2;
            border-radius: 14px;
            padding: 1.25rem;
            min-height: 145px;
            box-shadow: 0 4px 14px rgba(19, 57, 96, 0.06);
        }

        .info-card h3 {
            color: #0A3058;
            font-size: 1.05rem;
            margin-top: 0;
        }

        .info-card p {
            color: #506983;
            line-height: 1.55;
            margin-bottom: 0;
        }

        div.stButton {
            display: flex;
            justify-content: center;
            margin: 1.4rem 0;
        }

        div.stButton > button {
            width: 340px;
            min-height: 52px;
            border: none;
            border-radius: 9px;
            background: #0878B9;
            color: #FFFFFF !important;
            font-size: 1rem;
            font-weight: 700;
            white-space: nowrap;
        }

        div.stButton > button:hover {
            background: #075F91;
            color: #FFFFFF !important;
        }

        .result-card {
            background: #FFFFFF;
            border: 1px solid #DFE8F2;
            border-radius: 14px;
            padding: 1.3rem;
            min-height: 130px;
        }

        .result-label {
            color: #52708D;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }

        .result-value {
            color: #0A3058;
            font-size: 1.7rem;
            font-weight: 800;
            margin: 0.4rem 0;
        }

        [data-testid="stExpander"] {
            background-color: #FFFFFF;
            border: 1px solid #DFE8F2;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Sidebar: adjustable passenger parameters
with st.sidebar:
    st.header("🚢 Passenger Profile")
    st.write("Adjust the passenger details, then generate a prediction.")
    st.divider()

    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        index=2,
        help="1 = First class, 2 = Second class, 3 = Third class",
    )

    sex = st.selectbox(
        "Sex",
        options=["male", "female"],
        index=0,
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=22.0,
        step=1.0,
    )

    sibsp = st.number_input(
        "Siblings / Spouses Aboard",
        min_value=0,
        max_value=10,
        value=1,
        step=1,
    )

    parch = st.number_input(
        "Parents / Children Aboard",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
    )

    fare = st.number_input(
        "Ticket Fare",
        min_value=0.0,
        max_value=600.0,
        value=7.25,
        step=1.0,
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        options=["S", "C", "Q"],
        index=0,
        help="S = Southampton, C = Cherbourg, Q = Queenstown",
    )

    name = st.text_input(
        "Passenger Name",
        value="Smith, Mr. John",
    )

    cabin_known = st.checkbox(
        "Cabin number is available",
        value=False,
    )

    st.divider()
    st.caption("Scroll down in this panel to see all passenger inputs.")


cabin = "C123" if cabin_known else np.nan

passenger = {
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked,
    "Name": name,
    "Cabin": cabin,
}


# Main page header
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">MACHINE LEARNING • EXPLAINABLE AI</div>
        <div class="hero-title">🚢 Titanic Survival Predictor</div>
        <div class="hero-subtitle">
            Estimate a passenger's survival probability using passenger details,
            engineered features, a Random Forest classifier, and SHAP explanations.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Overview cards
card1, card2, card3 = st.columns(3)

with card1:
    st.markdown(
        """
        <div class="info-card">
            <h3>01. Adjust Parameters</h3>
            <p>Use the passenger form in the left sidebar to enter or change passenger details.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with card2:
    st.markdown(
        """
        <div class="info-card">
            <h3>02. Feature Engineering</h3>
            <p>The application derives passenger title, family size, and cabin availability.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with card3:
    st.markdown(
        """
        <div class="info-card">
            <h3>03. Explainable Result</h3>
            <p>SHAP helps show which features influenced the prediction for the passenger.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Prediction button
predict_clicked = st.button("Generate Survival Prediction", type="primary")


# Results and SHAP explanation
if predict_clicked:
    try:
        with st.spinner("Analysing passenger details..."):
            model, feature_cols = load_model_and_features()
            X = preprocess_one_passenger(passenger, feature_cols)

            predicted_class = int(model.predict(X)[0])
            survival_probability = float(model.predict_proba(X)[0, 1])

        st.subheader("Prediction Results")
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            outcome = "Survived ✅" if predicted_class == 1 else "Did Not Survive"
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted Outcome</div>
                    <div class="result-value">{outcome}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with result_col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Estimated Survival Probability</div>
                    <div class="result-value">{survival_probability:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.progress(survival_probability)

        st.subheader("Model Explanation")

        with st.expander("View SHAP feature explanation", expanded=True):
            st.write(
                "The chart ranks features by their contribution to this passenger's "
                "survival prediction."
            )

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values_survived = shap_values[0, :, 1]
            elif isinstance(shap_values, list):
                shap_values_survived = shap_values[1][0]
            else:
                shap_values_survived = shap_values[0]

            fig, ax = plt.subplots(figsize=(9, 5))
            shap.summary_plot(
                shap_values_survived.reshape(1, -1),
                X,
                plot_type="bar",
                max_display=10,
                show=False,
            )
            plt.title("Features Influencing the Survival Prediction")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    except FileNotFoundError as error:
        st.error(
            "Could not find the saved model files. Check that "
            "`models/titanic_model.pkl` and `models/feature_cols.pkl` exist."
        )
        st.exception(error)

    except Exception as error:
        st.error("The prediction could not be generated. Details:")
        st.exception(error)


# Project information
st.subheader("About the Project")

with st.expander("Model, data preparation, and explainability"):
    st.markdown(
        """
        This project uses a Random Forest classifier to estimate whether a
        passenger survived. The model uses passenger information such as class,
        sex, age, fare, family relationships, title, cabin availability, and
        embarkation port.

        Feature engineering creates a passenger title from the name, computes
        family size from sibling/spouse and parent/child counts, and records
        whether cabin information is available. SHAP provides an explanation
        of which features influenced the displayed prediction.

        This is an educational model demonstration, not a historical certainty
        about any individual passenger.
        """
    )

st.caption(
    "Titanic Survival Prediction | Random Forest • Feature Engineering • SHAP"
)