import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image

# Machine Learning Libraries
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from xgboost import XGBClassifier

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Zewail Student Performance AI",
    layout="wide",
    page_icon="🎓"
)

# --- 2. CSS STYLING (For the "Amazing" Look) ---
st.markdown("""
<style>
    .stButton>button {width: 100%; background-color: #0083B8; color: white; font-weight: bold; border-radius: 10px;}
    .reportview-container {background: #f5f5f5;}
    h1 {color: #003366;}
    h3 {color: #0083B8;}
</style>
""", unsafe_allow_html=True)

# --- 3. THE "BRAIN" (Training Function) ---
@st.cache_resource
def get_model_pipeline():
    # Checks if data exists
    if not os.path.exists('Term_Project_Dataset_20K.csv'):
        st.error("❌ Critical Error: 'Term_Project_Dataset_20K.csv' not found! Please put the CSV file in this folder.")
        st.stop()
        
    with st.spinner('🧠 Training AI Models from scratch (including Winsorization)... Please wait...'):
        # A. Load Data
        df = pd.read_csv('Term_Project_Dataset_20K.csv')
        
        # [cite_start]B. Clean Targets (Drop rows where Score/Grade is missing) [cite: 16]
        df = df.dropna(subset=['final_score', 'final_grade', 'pass_fail'])

        # C. Separate Features and Targets
        targets = ['final_score', 'final_grade', 'pass_fail']
        X = df.drop(columns=targets)
        y_score = df['final_score']
        y_grade = df['final_grade']
        y_pf = df['pass_fail']
        
        # D. Identify Columns
        cat_cols = ['gender', 'part_time_job', 'course_type']
        # Select all numbers except the targets we dropped
        num_cols = [c for c in X.columns if c not in cat_cols]
        
        # --- PREPROCESSING STEPS (MATCHING NOTEBOOK) ---
        
        # [cite_start]1. Impute Missing Values [cite: 457]
        imputer_num = SimpleImputer(strategy='median')
        X[num_cols] = imputer_num.fit_transform(X[num_cols])
        
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])
        
        # [cite_start]2. Winsorization (Outlier Capping) [cite: 489-502]
        # We cap specific columns at 1st and 99th percentiles to handle extreme outliers
        outlier_cols = ['parent_income', 'online_portal_usage_minutes', 
                        'lecture_attendance_rate', 'stress_level', 'sleep_hours']
        
        winsor_stats = {} # We must save these stats to apply to new users!
        
        for col in outlier_cols:
            if col in X.columns:
                lower = np.percentile(X[col], 1)
                upper = np.percentile(X[col], 99)
                X[col] = X[col].clip(lower=lower, upper=upper)
                winsor_stats[col] = (lower, upper)

        # [cite_start]3. Scaling (StandardScaler) [cite: 518]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[num_cols])
        
        # [cite_start]4. Encoding (OneHot) [cite: 506]
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_encoded = encoder.fit_transform(X[cat_cols])
        
        # 5. Combine into Final Training Data
        feat_names = encoder.get_feature_names_out(cat_cols)
        X_final = pd.DataFrame(np.hstack([X_scaled, X_encoded]), 
                               columns=list(num_cols) + list(feat_names))
        
        # --- MODEL TRAINING ---
        
        # [cite_start]Model 1: Final Score (Ridge) [cite: 93]
        m_score = Ridge()
        m_score.fit(X_final, y_score)
        
        # [cite_start]Model 2: Pass/Fail (XGBoost) [cite: 1589]
        # Map targets to 1/0
        y_pf_map = y_pf.map({'Pass': 1, 'Fail': 0})
        m_pf = XGBClassifier(eval_metric='logloss')
        m_pf.fit(X_final, y_pf_map)
        
        # [cite_start]Model 3: Final Grade (XGBoost) [cite: 1962]
        le = LabelEncoder()
        y_grade_map = le.fit_transform(y_grade)
        m_grade = XGBClassifier(eval_metric='mlogloss')
        m_grade.fit(X_final, y_grade_map)
        
        # Return everything we need to process new data
        return m_score, m_pf, m_grade, scaler, encoder, le, num_cols, cat_cols, winsor_stats

# --- 4. LOAD THE PIPELINE ---
try:
    model_score, model_pf, model_grade, scaler, encoder, label_enc, NUM_COLS, CAT_COLS, WINSOR_STATS = get_model_pipeline()
    st.toast("System Ready! AI successfully trained.", icon="✅")
except Exception as e:
    st.error(f"Training Failed: {e}")
    st.stop()

# --- 5. APP HEADER & LOGO ---
col_logo, col_title = st.columns([1, 4])

with col_logo:
    # Try to load local 'logo.png', otherwise use a web fallback
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)
    else:
        # Fallback if you forget to put the file there
        st.image("https://upload.wikimedia.org/wikipedia/en/e/e0/Zewail_City_of_Science_and_Technology_Logo.png", width=100)

with col_title:
    st.title("🎓 Smart Student Performance Prediction System")
    st.markdown("### Zewail City - CIE 417 Machine Learning Project")

st.markdown("---")

# --- 6. SIDEBAR INPUTS (38 Features) ---
st.sidebar.header("📝 Student Data Input")

with st.form("prediction_form"):
    
    # [cite_start]Group A: Demographic [cite: 25]
    with st.expander("👤 A. Demographic Information", expanded=True):
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 15, 50, 20)
        gender = c2.selectbox("Gender", ["Male", "Female"])
        income = c1.number_input("Parent Income", 0, 1000000, 50000)
        siblings = c2.number_input("Num Siblings", 0, 10, 2)
        support = c1.slider("Family Support (0-5)", 0, 5, 3)
        commute = c2.number_input("Commute Time (min)", 0, 180, 30)
        job = c1.selectbox("Part-time Job", ["No", "Yes"]) 

    # [cite_start]Group B: Academic History [cite: 32]
    with st.expander("📚 B. Academic History"):
        c1, c2 = st.columns(2)
        gpa = c1.number_input("Previous GPA", 0.0, 4.0, 3.0, step=0.01)
        fail = c2.number_input("Failed Courses", 0, 10, 0)
        hs_g = c1.number_input("High School Grade", 0, 100, 85)
        warn = c2.number_input("Academic Warnings", 0, 5, 0)
        att = c1.slider("Past Attendance Rate %", 0, 100, 90)
        study = c2.number_input("Study Hours (Last Sem)", 0, 500, 100)
        cred = c1.number_input("Prior Sem Credits", 0, 30, 15)
        
        st.caption("Background Scores")
        b1, b2, b3 = st.columns(3)
        math = b1.number_input("Math Score", 0, 100, 70)
        lang = b2.number_input("Lang Score", 0, 100, 70)
        sci = b3.number_input("Science Score", 0, 100, 70)

    # [cite_start]Group C: Behavioral [cite: 43]
    with st.expander("⚡ C. Behavioral & Engagement"):
        c1, c2 = st.columns(2)
        lec = c1.slider("Lecture Attendance %", 0, 100, 85)
        sub = c2.slider("Assignment Submission %", 0, 100, 90)
        lab = c1.slider("Lab Participation %", 0, 100, 80)
        port = c2.number_input("Portal Usage (min)", 0, 10000, 500)
        
        c3, c4 = st.columns(2)
        mid = c3.number_input("Midterm Score", 0, 100, 75)
        quiz = c4.number_input("Quiz Avg Score", 0, 100, 75)
        
        post = c1.number_input("Forum Posts", 0, 100, 5)
        visit = c2.number_input("Library Visits/Month", 0, 30, 4)
        grp = c1.number_input("Group Project Activity", 0, 100, 50)
        late = c2.number_input("Lateness Count", 0, 20, 0)

    # [cite_start]Group D: Psychological [cite: 54]
    with st.expander("🧠 D. Psychological Factors"):
        c1, c2 = st.columns(2)
        strs = c1.slider("Stress Level (0-10)", 0, 10, 5)
        mot = c2.slider("Motivation Level (0-10)", 0, 10, 7)
        conc = c1.slider("Concentration Level (0-10)", 0, 10, 6)
        anx = c2.slider("Exam Anxiety (0-10)", 0, 10, 4)
        slp = c1.number_input("Sleep Hours", 0.0, 12.0, 7.0)
        swk = c2.number_input("Study Time/Week (hrs)", 0, 100, 20)

    # [cite_start]Group E: Institutional [cite: 65]
    with st.expander("🏫 E. Institutional Data"):
        c1, c2 = st.columns(2)
        diff = c1.slider("Course Difficulty (1-5)", 1, 5, 3)
        exp = c2.number_input("Teacher Experience (yrs)", 0, 40, 10)
        size = c1.number_input("Class Size", 10, 500, 50)
        pre = c2.number_input("Num Prerequisites", 0, 5, 1)
        ctype = c1.selectbox("Course Type", ["Mandatory", "Elective"]) 

    submit = st.form_submit_button("🚀 Predict Performance")

# --- 7. PREDICTION LOGIC ---
if submit:
    # 1. Create Raw DataFrame from Inputs
    data = {
        'age': age, 'parent_income': income, 'num_siblings': siblings,
        'family_support': support, 'commute_time_min': commute,
        'previous_gpa': gpa, 'num_failed_courses': fail,
        'high_school_grade': hs_g, 'math_background_score': math,
        'language_background_score': lang, 'science_background_score': sci,
        'prior_semester_credits': cred, 'study_hours_last_semester': study,
        'past_attendance_rate': att, 'academic_warnings_count': warn,
        'lecture_attendance_rate': lec, 'assignment_submission_rate': sub,
        'quiz_avg_score': quiz, 'midterm_score': mid,
        'lab_participation_rate': lab, 'online_portal_usage_minutes': port,
        'group_project_activity': grp, 'library_visits_per_month': visit,
        'discussion_forum_posts': post, 'lateness_count': late,
        'stress_level': strs, 'sleep_hours': slp, 'motivation_level': mot,
        'study_time_per_week': swk, 'concentration_level': conc,
        'exam_anxiety_level': anx, 'course_difficulty_rating': diff,
        'teacher_experience_years': exp, 'class_size': size,
        'num_prerequisites': pre,
        # Categorical
        'gender': gender, 'part_time_job': job, 'course_type': ctype
    }
    
    df_new = pd.DataFrame([data])
    
    try:
        # 2. Apply Winsorization (Using saved stats)
        for col, (lower, upper) in WINSOR_STATS.items():
            if col in df_new.columns:
                 df_new[col] = df_new[col].clip(lower=lower, upper=upper)

        # 3. Apply Scaling
        df_new[NUM_COLS] = scaler.transform(df_new[NUM_COLS])
        
        # 4. Apply Encoding
        cats_encoded = encoder.transform(df_new[CAT_COLS])
        feat_names = encoder.get_feature_names_out(CAT_COLS)
        
        # 5. Reconstruct DataFrame
        X_live = pd.DataFrame(
            np.hstack([df_new[NUM_COLS], cats_encoded]),
            columns=list(NUM_COLS) + list(feat_names)
        )
        
        # 6. Make Predictions
        st.divider()
        st.subheader("📊 Prediction Results")
        
        c1, c2, c3 = st.columns(3)
        
        # Result 1: Final Score
        pred_score = model_score.predict(X_live)[0]
        c1.metric("Predicted Final Score", f"{pred_score:.1f} / 100")
        c1.progress(min(int(pred_score), 100))
        
        # Result 2: Pass/Fail
        pred_pf_idx = model_pf.predict(X_live)[0]
        pf_status = "PASS" if pred_pf_idx == 1 else "FAIL"
        pf_color = "green" if pf_status == "PASS" else "red"
        
        c2.markdown("### Pass/Fail Status")
        c2.markdown(f"<h1 style='color:{pf_color};'>{pf_status}</h1>", unsafe_allow_html=True)
        
        # Result 3: Grade
        pred_grade_idx = model_grade.predict(X_live)[0]
        pred_grade = label_enc.inverse_transform([pred_grade_idx])[0]
        c3.metric("Predicted Letter Grade", pred_grade)
        
        st.success("Analysis Complete!")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Ensure the inputs are within valid ranges.")