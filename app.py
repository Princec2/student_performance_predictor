import streamlit as st
import joblib
import pandas as pd

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)
# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Main title */
    h1 {
        text-align: center;
        font-size: 42px !important;
        font-weight: 700 !important;
        margin-bottom: 10px;
    }

    /* Subtitle */
    .stApp p {
        font-size: 16px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 45px;
        font-size: 16px;
        font-weight: 600;
    }

    /* Input boxes */
    div[data-testid="stNumberInput"] {
        margin-bottom: 10px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }
    /* Analytics section */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }
    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)

HISTORY_FILE = "prediction_history.csv"

# --------------------------------------------------
# LOAD PREDICTION HISTORY
# --------------------------------------------------

if "history" not in st.session_state:
    try:
        st.session_state.history = pd.read_csv(
            HISTORY_FILE
        ).to_dict("records")

    except FileNotFoundError:
        st.session_state.history = []

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

model = joblib.load("student_performance_model.pkl")

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🎓 Student Performance Predictor")

st.write(
    "Enter student details to predict final marks."
)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

col1, col2 = st.columns(2)

# LEFT COLUMN
with col1:

    study_hours = st.number_input(
        "Study Hours",
        min_value=1.0,
        max_value=10.0,
        value=5.0
    )

    previous_marks = st.number_input(
        "Previous Marks",
        min_value=50.0,
        max_value=100.0,
        value=60.0
    )

# RIGHT COLUMN
with col2:

    attendance = st.number_input(
        "Attendance (%)",
        min_value=40.0,
        max_value=95.0,
        value=75.0
    )

    assignment_score = st.number_input(
        "Assignment Score",
        min_value=40.0,
        max_value=100.0,
        value=70.0
    )

# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

if st.button(
    "🎯 Predict Final Marks",
    use_container_width=True
):

    # Create input DataFrame
    student = pd.DataFrame(
        [[
            study_hours,
            attendance,
            previous_marks,
            assignment_score
        ]],
        columns=[
            "study_hours",
            "attendance",
            "previous_marks",
            "assignment_score"
        ]
    )

    # Make prediction
    prediction = model.predict(student)

    # Keep prediction between 0 and 100
    result = max(
        0,
        min(
            100,
            float(prediction[0])
        )
    )

    # Save prediction in session history
    st.session_state.history.append({

        "Study Hours": study_hours,

        "Attendance": attendance,

        "Previous Marks": previous_marks,

        "Assignment Score": assignment_score,

        "Predicted Marks": round(result, 2)
    })

    # Save history to CSV
    history_df = pd.DataFrame(
        st.session_state.history
    )

    history_df.to_csv(
        HISTORY_FILE,
        index=False
    )

# --------------------------------------------------
# PREDICTION RESULT CARD
# --------------------------------------------------
    st.subheader("🎯 Prediction Result")

    st.success(f"Predicted Final Marks: {result:.2f} / 100")

    st.progress(int(result))

    # --------------------------------------------------
    # GRADE & PERFORMANCE RECOMMENDATION
    # --------------------------------------------------

    if result >= 90:
        grade = "A+"
        message = "🌟 Outstanding performance! Keep it up."

    elif result >= 80:
        grade = "A"
        message = "👏 Excellent performance!"

    elif result >= 70:
        grade = "B"
        message = "👍 Good performance. Keep improving."

    elif result >= 60:
        grade = "C"
        message = "📚 Average performance. More practice can help."

    elif result >= 50:
        grade = "D"
        message = "⚠️ Needs improvement. Focus more on studies."

    else:
        grade = "F"
        message = "📖 More preparation is recommended."

    st.info(
        f"🎓 Grade: {grade}\n\n{message}"
    )
    


# --------------------------------------------------
# ANALYTICS DASHBOARD
# --------------------------------------------------

st.markdown("---")

st.subheader("📊 Analytics Dashboard")

if len(st.session_state.history) > 0:

    analytics_df = pd.DataFrame(
        st.session_state.history
    )

    # Basic analytics
    total_predictions = len(analytics_df)

    average_marks = analytics_df[
        "Predicted Marks"
    ].mean()

    highest_marks = analytics_df[
        "Predicted Marks"
    ].max()

    lowest_marks = analytics_df[
        "Predicted Marks"
    ].min()

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:

        st.metric(
            "Total Predictions",
            total_predictions
        )

    with metric2:

        st.metric(
            "Average Marks",
            f"{average_marks:.2f}"
        )

    with metric3:

        st.metric(
            "Highest Marks",
            f"{highest_marks:.2f}"
        )

    with metric4:

        st.metric(
            "Lowest Marks",
            f"{lowest_marks:.2f}"
        )

    # --------------------------------------------------
    # LATEST PREDICTION
    # --------------------------------------------------

    last_prediction = st.session_state.history[-1]

    st.markdown("---")

    st.subheader("📈 Marks Comparison")

    chart_data = pd.DataFrame(
        {
            "Marks": [
                last_prediction["Previous Marks"],
                last_prediction["Assignment Score"],
                last_prediction["Predicted Marks"]
            ]
        },
        index=[
            "Previous Marks",
            "Assignment Score",
            "Predicted Final Marks"
        ]
    )

    st.bar_chart(chart_data)

    # --------------------------------------------------
    # LATEST PREDICTION RESULT
    # --------------------------------------------------

    st.subheader("🎯 Latest Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric(
            "Previous Marks",
            f"{last_prediction['Previous Marks']:.1f}"
        )

    with result_col2:

        st.metric(
            "Predicted Marks",
            f"{last_prediction['Predicted Marks']:.2f}"
        )

    with result_col3:

        st.metric(
            "Assignment Score",
            f"{last_prediction['Assignment Score']:.1f}"
        )

    # --------------------------------------------------
    # PERFORMANCE STATUS
    # --------------------------------------------------

    predicted_marks = last_prediction[
        "Predicted Marks"
    ]

    if predicted_marks > 80:

        st.success(
            "🌟 Excellent Performance!"
        )

    elif predicted_marks >= 60:

        st.info(
            "👍 Good Performance!"
        )

    elif predicted_marks >= 40:

        st.warning(
            "📚 Needs Improvement"
        )

    else:

        st.error(
            "⚠️ At Risk"
        )

    # --------------------------------------------------
    # PREDICTION HISTORY
    # --------------------------------------------------

    st.markdown("---")

    st.subheader("📋 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No predictions yet. Enter student details and click Predict Final Marks."
    )

# --------------------------------------------------
# CLEAR HISTORY
# --------------------------------------------------

st.markdown("---")

if st.button(
    "🗑️ Clear Prediction History",
    use_container_width=True
):

    # Clear session history
    st.session_state.history = []

    # Create empty CSV
    pd.DataFrame(
        columns=[
            "Study Hours",
            "Attendance",
            "Previous Marks",
            "Assignment Score",
            "Predicted Marks"
        ]
    ).to_csv(
        HISTORY_FILE,
        index=False
    )

    st.rerun()

# --------------------------------------------------
# DOWNLOAD HISTORY
# --------------------------------------------------

history_df = pd.DataFrame(
    st.session_state.history
)

csv_data = history_df.to_csv(
    index=False
)

st.download_button(
    label="📥 Download Prediction History",
    data=csv_data,
    file_name="student_prediction_history.csv",
    mime="text/csv",
    use_container_width=True
)

# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

st.markdown("---")

st.subheader("📘 About This Project")

st.write(
    "This application uses a Machine Learning model "
    "to predict a student's final marks based on "
    "study hours, attendance, previous marks, "
    "and assignment score."
)

st.caption(
    "🎓 Student Performance Prediction • "
    "Built with Python, Pandas, Scikit-learn & Streamlit"
)
st.caption(
    "Version 1.0 • Student Performance Predictor"
)


