
import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score)

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, "src"))
from features import clean_and_engineer  # noqa: E402


def path(*parts):
    return os.path.join(BASE, *parts)


MODEL_PATH = path("models", "credit_model.joblib")
CONFIG_PATH = path("models", "config.json")
RESULTS_PATH = path("models", "results.csv")
SCORES_PATH = path("models", "test_scores.csv")

# EDIT THESE so the app describes YOUR project
MODEL_DESCRIPTION = "Random Forest (scikit-learn), tuned with randomized search and cross-validation"
INSIGHTS = [
    "Default rate falls as age rises",
    "Late-payment counts are the strongest signal",
    "income, debt_ratio, dependents, and open_lines all show negligible linear correlation with each other and with default",
    "utlization and debt are strong markers",
]

# The 10 raw input columns, in the order the Kaggle file uses.
FEATURES = ["utilization", "age", "late_30_59", "debt_ratio", "income",
            "open_lines", "late_90", "real_estate_loans", "late_60_89", "dependents"]

KAGGLE_RENAME = {
    "SeriousDlqin2yrs": "default",
    "RevolvingUtilizationOfUnsecuredLines": "utilization",
    "NumberOfTime30-59DaysPastDueNotWorse": "late_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "income",
    "NumberOfOpenCreditLinesAndLoans": "open_lines",
    "NumberOfTimes90DaysLate": "late_90",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "late_60_89",
    "NumberOfDependents": "dependents",
}

LABELS = {
    "age": "Age",
    "income": "Monthly income",
    "debt_ratio": "Debt ratio",
    "utilization": "Credit utilization",
    "open_lines": "Open credit lines and loans",
    "real_estate_loans": "Real-estate loans",
    "dependents": "Dependents",
    "late_30_59": "Times 30-59 days late",
    "late_60_89": "Times 60-89 days late",
    "late_90": "Times 90+ days late",
}

DEFAULTS = dict(age=40, income=5000, income_unknown=False, debt_ratio=0.30,
                utilization=0.30, open_lines=8, real_estate_loans=1, dependents=0,
                late_30_59=0, late_60_89=0, late_90=0)
SAFE_PROFILE = dict(DEFAULTS, age=52, income=9000, debt_ratio=0.15,
                    utilization=0.05, open_lines=9, dependents=1)
RISKY_PROFILE = dict(DEFAULTS, age=27, income=2200, debt_ratio=0.90,
                     utilization=1.10, open_lines=12, real_estate_loans=0,
                     dependents=3, late_30_59=3, late_60_89=2, late_90=4)

# Fields that get a slider + a number box, with their range and step.
# int fields use int bounds/step; float fields use float bounds/step -
# Streamlit requires the types to match within one widget.
NUMERIC_FIELDS = {
    "age": dict(min_value=18, max_value=100, step=1),
    "debt_ratio": dict(min_value=0.0, max_value=5.0, step=0.01),
    "utilization": dict(min_value=0.0, max_value=2.0, step=0.01),
    "open_lines": dict(min_value=0, max_value=40, step=1),
    "real_estate_loans": dict(min_value=0, max_value=10, step=1),
    "dependents": dict(min_value=0, max_value=10, step=1),
    "late_30_59": dict(min_value=0, max_value=10, step=1),
    "late_60_89": dict(min_value=0, max_value=10, step=1),
    "late_90": dict(min_value=0, max_value=10, step=1),
}

# Palette: ink, teal for "fine", brick red for "risky"
INK, TEAL, RED = "#17212b", "#0f766e", "#b42318"
TEAL_SOFT, RED_SOFT = "#cfe8e4", "#f6d5d1"

st.set_page_config(page_title="Credit risk predictor", layout="wide")
st.markdown(
    f"""
    <style>
    .verdict {{ padding: 1rem 1.25rem; border-left: 6px solid {INK}; background: rgba(128,128,128,.08); }}
    .verdict.low  {{ border-left-color: {TEAL}; }}
    .verdict.high {{ border-left-color: {RED}; }}
    .verdict h3 {{ margin: 0 0 .25rem 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- loading
if not (os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH)):
    st.error("Model files not found. Save `models/credit_model.joblib` and "
             "`models/config.json` from your modeling notebook, then reload.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_threshold():
    with open(CONFIG_PATH) as f:
        return float(json.load(f)["threshold"])


@st.cache_data
def load_csv(p):
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data
def pr_curve(y_true, proba):
    grid = np.linspace(0.02, 0.98, 49)
    prec = [precision_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    rec = [recall_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    return grid, prec, rec


model = load_model()
saved_threshold = load_threshold()


def score(df_raw: pd.DataFrame) -> np.ndarray:
    """df_raw holds the 10 raw columns; returns risk scores between 0 and 1."""
    engineered = clean_and_engineer(df_raw[FEATURES])
    cols = list(getattr(model, "feature_names_in_", engineered.columns))
    return model.predict_proba(engineered[cols])[:, 1]


def build_row(p) -> dict:
    return {
        "utilization": p["utilization"], "age": p["age"],
        "late_30_59": p["late_30_59"], "debt_ratio": p["debt_ratio"],
        "income": np.nan if p["income_unknown"] else p["income"],
        "open_lines": p["open_lines"], "late_90": p["late_90"],
        "real_estate_loans": p["real_estate_loans"], "late_60_89": p["late_60_89"],
        "dependents": p["dependents"],
    }


def gauge(risk, threshold):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Risk score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": INK},
            "steps": [{"range": [0, threshold * 100], "color": TEAL_SOFT},
                      {"range": [threshold * 100, 100], "color": RED_SOFT}],
            "threshold": {"line": {"color": RED, "width": 4},
                          "thickness": 0.85, "value": threshold * 100},
        }))
    fig.update_layout(height=290, margin=dict(l=20, r=20, t=60, b=10))
    return fig


# ----------------------------------------------------------- session state
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
for key in NUMERIC_FIELDS:
    # The number box next to each slider lives in its own "<key>_num" slot,
    # seeded once from the canonical value so the two start in agreement.
    st.session_state.setdefault(f"{key}_num", st.session_state[key])


def load_profile(profile):
    for key, value in profile.items():
        st.session_state[key] = value
        if key in NUMERIC_FIELDS:
            st.session_state[f"{key}_num"] = value


def _sync_from_num(key):
    """Number box edited -> copy its value into the slider's key."""
    st.session_state[key] = st.session_state[f"{key}_num"]


def _sync_from_slider(key):
    """Slider dragged -> copy its value into the number box's key."""
    st.session_state[f"{key}_num"] = st.session_state[key]


def slider_with_box(label, key, help=None):
    """Render a slider and a type-in number box for the same value, kept in sync."""
    cfg = NUMERIC_FIELDS[key]
    col_slider, col_num = st.columns([3, 1])
    with col_slider:
        st.slider(label, cfg["min_value"], cfg["max_value"], step=cfg["step"],
                  key=key, help=help, on_change=_sync_from_slider, args=(key,))
    with col_num:
        st.number_input(label, cfg["min_value"], cfg["max_value"], step=cfg["step"],
                        key=f"{key}_num", on_change=_sync_from_num, args=(key,),
                        label_visibility="collapsed")


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Decision threshold")
    threshold = st.slider(
        "Flag as high risk at or above", 0.05, 0.95,
        float(np.clip(saved_threshold, 0.05, 0.95)), 0.01,
        help="Lower = catch more defaulters but raise more false alarms. "
             "Higher = fewer false alarms but more missed defaulters.")
    st.caption(f"Threshold chosen during model development: {saved_threshold:.2f}")
    st.divider()
    st.caption(MODEL_DESCRIPTION)

st.title("Credit risk predictor")
st.write("Enter a financial profile to see how likely a model trained on past "
         "borrowers thinks it is to lead to serious delinquency (90+ days late "
         "within two years). This is a learning project, not financial advice.")

tab_predict, tab_perf, tab_data, tab_about = st.tabs(
    ["Predict", "Model performance", "Data insights", "About"])

# ------------------------------------------------------------ tab: predict
with tab_predict:
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.subheader("Applicant profile")
        b1, b2, b3 = st.columns(3)
        b1.button("Load a safe profile", on_click=load_profile, args=(SAFE_PROFILE,))
        b2.button("Load a risky profile", on_click=load_profile, args=(RISKY_PROFILE,))
        b3.button("Reset", on_click=load_profile, args=(DEFAULTS,))

        c1, c2 = st.columns(2)
        with c1:
            slider_with_box("Age", "age")
            slider_with_box("Debt ratio", "debt_ratio",
                            help="Monthly debt payments divided by monthly income.")
            slider_with_box("Credit utilization", "utilization",
                            help="Card and credit-line balances divided by limits. 1.0 means fully used.")
            slider_with_box("Open credit lines and loans", "open_lines")
            slider_with_box("Real-estate loans", "real_estate_loans")
        with c2:
            st.checkbox("I don't know the monthly income", key="income_unknown")
            st.number_input("Monthly income", min_value=0, max_value=1_000_000,
                            step=500, key="income",
                            disabled=st.session_state["income_unknown"])
            slider_with_box("Dependents", "dependents")
            slider_with_box("Times 30-59 days late", "late_30_59")
            slider_with_box("Times 60-89 days late", "late_60_89")
            slider_with_box("Times 90+ days late", "late_90")

    row = pd.DataFrame([build_row(st.session_state)])
    risk = float(score(row)[0])
    is_high = risk >= threshold

    with right:
        st.subheader("Result")
        st.plotly_chart(gauge(risk, threshold))
        if is_high:
            st.markdown(
                '<div class="verdict high"><h3>High risk</h3>'
                'The score is at or above the threshold, so the model flags this '
                'profile as likely to become seriously delinquent.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="verdict low"><h3>Low risk</h3>'
                'The score is below the threshold, so the model does not flag '
                'this profile.</div>',
                unsafe_allow_html=True)

        flags = []
        if st.session_state["utilization"] > 1:
            flags.append("Credit utilization is above 100%.")
        if st.session_state["late_90"] > 0:
            flags.append("There are 90+ day late payments on record.")
        if st.session_state["late_30_59"] + st.session_state["late_60_89"] > 2:
            flags.append("Several shorter late payments are recorded.")
        if st.session_state["income_unknown"]:
            flags.append("Income is unknown, so the model fills it with a typical value.")
        if flags:
            st.markdown("**Worth noticing**")
            for f in flags:
                st.markdown(f"- {f}")
        st.caption("The score is the model's output, not a calibrated probability: "
                   "the model was trained with class weights, so read it as a ranking "
                   "where higher means riskier.")

    with st.expander("What-if analysis: change one input and watch the score"):
        ranges = {
            "utilization": np.linspace(0, 2, 41),
            "debt_ratio": np.linspace(0, 5, 41),
            "age": np.arange(18, 91, 2),
            "income": np.linspace(500, 20000, 40),
            "late_90": np.arange(0, 11),
            "late_60_89": np.arange(0, 11),
            "late_30_59": np.arange(0, 11),
            "open_lines": np.arange(0, 41),
            "real_estate_loans": np.arange(0, 11),
            "dependents": np.arange(0, 11),
        }
        feature = st.selectbox("Input to vary", list(ranges), format_func=LABELS.get)
        values = ranges[feature]
        grid = pd.concat([row] * len(values), ignore_index=True)
        grid[feature] = values
        curve = score(grid) * 100
        fig = go.Figure(go.Scatter(x=values, y=curve, mode="lines",
                                   line=dict(color=TEAL, width=3), name="Risk score"))
        fig.add_hline(y=threshold * 100, line_dash="dash", line_color=RED,
                      annotation_text="threshold")
        current = row[feature].iloc[0]
        if pd.notna(current):
            fig.add_trace(go.Scatter(x=[current], y=[risk * 100], mode="markers",
                                     marker=dict(size=12, color=INK), name="Current profile"))
        fig.update_layout(height=340, xaxis_title=LABELS[feature],
                          yaxis_title="Risk score (%)", margin=dict(t=20))
        st.plotly_chart(fig)
        st.caption("All other inputs stay at the values you set above.")

    with st.expander("Batch scoring: upload a CSV"):
        template = pd.DataFrame([build_row(SAFE_PROFILE), build_row(RISKY_PROFILE)])[FEATURES]
        st.download_button("Download a CSV template", template.to_csv(index=False),
                           "template.csv", "text/csv")
        upload = st.file_uploader(
            "CSV with the 10 input columns (short names or the original Kaggle names)",
            type="csv")
        if upload is not None:
            data = pd.read_csv(upload).rename(columns=KAGGLE_RENAME)
            data = data.drop(columns=[c for c in ["Unnamed: 0", "default"] if c in data.columns])
            missing = [c for c in FEATURES if c not in data.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}. Use the template for the exact names.")
            else:
                scores = score(data)
                out = data[FEATURES].copy()
                out["risk_score"] = scores.round(4)
                out["decision"] = np.where(scores >= threshold, "High risk", "Low risk")
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows scored", f"{len(out):,}")
                m2.metric("Flagged high risk", f"{int((scores >= threshold).sum()):,}")
                m3.metric("Share flagged", f"{(scores >= threshold).mean():.1%}")
                st.dataframe(out.head(500))
                st.download_button("Download results", out.to_csv(index=False),
                                   "predictions.csv", "text/csv")

# ------------------------------------------------- tab: model performance
with tab_perf:
    st.subheader("How well does the model work?")
    results = load_csv(RESULTS_PATH)
    if results is not None:
        st.markdown("**Model comparison** (cross-validation on the training set)")
        st.dataframe(results, hide_index=True)
    else:
        st.info("Save your comparison table from the notebook as `models/results.csv` to show it here.")

    scores_df = load_csv(SCORES_PATH)
    if scores_df is not None:
        y_true = scores_df["y_true"].to_numpy()
        proba = scores_df["proba"].to_numpy()
        pred = (proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)

        st.markdown("**Held-out test set** - move the threshold in the sidebar to see the trade-off")
        m = st.columns(5)
        m[0].metric("ROC-AUC", f"{roc_auc_score(y_true, proba):.3f}")
        m[1].metric("Precision", f"{prec:.3f}")
        m[2].metric("Recall", f"{rec:.3f}")
        m[3].metric("F1-score", f"{f1_score(y_true, pred, zero_division=0):.3f}")
        m[4].metric("Threshold", f"{threshold:.2f}")
        st.write(f"At this threshold the model catches {rec:.0%} of the borrowers who really "
                 f"defaulted, and {1 - prec:.0%} of the people it flags did not default.")

        g1, g2 = st.columns(2)
        z = [[tn, fp], [fn, tp]]
        cm = go.Figure(go.Heatmap(
            z=z, x=["Predicted low risk", "Predicted high risk"],
            y=["Actually fine", "Actually defaulted"], text=z,
            texttemplate="%{text:,}", colorscale="Teal", showscale=False))
        cm.update_yaxes(autorange="reversed")
        cm.update_layout(height=340, title="Confusion matrix", margin=dict(t=50))
        g1.plotly_chart(cm)

        grid_t, prec_curve, rec_curve = pr_curve(y_true, proba)
        pr = go.Figure()
        pr.add_trace(go.Scatter(x=grid_t, y=prec_curve, name="Precision", line=dict(color=TEAL, width=3)))
        pr.add_trace(go.Scatter(x=grid_t, y=rec_curve, name="Recall", line=dict(color=RED, width=3)))
        pr.add_vline(x=threshold, line_dash="dash", line_color=INK)
        pr.update_layout(height=340, title="Precision and recall by threshold",
                         xaxis_title="Threshold", margin=dict(t=50))
        g2.plotly_chart(pr)
    else:
        st.info("Save the test-set scores from the notebook as `models/test_scores.csv` "
                "(columns `y_true`, `proba`) to unlock the live metrics.")

    img_cols = st.columns(2)
    for col, (caption, file) in zip(img_cols, [("ROC curves", "roc_curve.png"),
                                               ("Feature importance", "feature_importance.png")]):
        if os.path.exists(path("assets", file)):
            col.image(path("assets", file), caption=caption)

# ------------------------------------------------------- tab: data insights
with tab_data:
    st.subheader("What the data shows")
    for item in INSIGHTS:
        st.markdown(f"- {item}")
    images = sorted(glob.glob(path("assets", "eda_*.png")))
    if images:
        cols = st.columns(2)
        for i, img in enumerate(images):
            cols[i % 2].image(img)
    else:
        st.info("Save your best EDA charts as `assets/eda_*.png` to show them here.")

# --------------------------------------------------------------- tab: about
with tab_about:
    st.subheader("How this works")
    st.markdown(
        "**Data.** The *Give Me Some Credit* dataset from Kaggle: about 150,000 borrowers, "
        "with roughly 7% who became 90+ days delinquent within two years.\n\n"
        "**Steps.** Clean the data, engineer features such as total late payments and income "
        "per person, train and compare Logistic Regression, Decision Tree and Random Forest "
        "models, tune the best one, then choose a threshold that favours catching defaulters.\n\n"
        f"**Final model.** {MODEL_DESCRIPTION}."
    )
    with st.expander("What do the metrics mean?"):
        st.markdown(
            "- **Precision:** of the people flagged as high risk, the share who really defaulted.\n"
            "- **Recall:** of the people who really defaulted, the share the model flagged.\n"
            "- **F1-score:** a single number that balances precision and recall.\n"
            "- **ROC-AUC:** how well the model ranks risky borrowers above safe ones, "
            "regardless of threshold. 0.5 is random guessing and 1.0 is perfect."
        )
    st.subheader("Limitations")
    st.markdown(
        "- The data is old and comes from one lender, so it may not reflect other populations.\n"
        "- The model has not been audited for fairness across groups. A real lender must do this.\n"
        "- The score is a ranking, not a calibrated probability.\n"
        "- This is an educational project and must not be used for real lending decisions."
    )