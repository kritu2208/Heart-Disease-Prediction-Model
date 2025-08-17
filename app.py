import streamlit as st
import pickle
import pandas as pd
import numpy as np

# A function to load the model
@st.cache_resource
def load_model():
    """
    Loads the pre-trained Gradient Boosting model from the .pkl file using pickle.
    """
    try:
        # We are now loading the new, converted file
        with open('cardio_compatible_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Error: The model file 'cardio_compatible_model.pkl' was not found. Please ensure it is in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

# Load the model
model = load_model()

# Set up the Streamlit app layout with a wider view
st.set_page_config(
    page_title="Cardiovascular Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS for a more beautiful, medical-themed UI with a darker background
st.markdown("""
<style>
    .reportview-container .main {
        color: #e2e2e2;
        background-color: #2e4053;
    }
    .stApp {
        background-color: #2e4053;
    }
    h1, h2, h3 {
        color: #d6eaf8;
    }
    .stButton>button {
        background-color: #2e86c1;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        border: none;
        padding: 10px 24px;
        font-size: 18px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1a5276;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .stProgress > div > div > div > div {
        background-color: #28a745; /* Default for low risk */
    }
    .stProgress.high-risk > div > div > div > div {
        background-color: #dc3545; /* Red for high risk */
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .prediction-high {
        background-color: #dc3545;
        border: 2px solid #dc3545;
        color: white;
    }
    .prediction-low {
        background-color: #28a745;
        border: 2px solid #28a745;
        color: white;
    }
    .st-expanderHeader {
        font-size: 1.25rem;
        font-weight: 600;
        color: #d6eaf8;
    }
    label {
        color: #d6eaf8;
    }
</style>
""", unsafe_allow_html=True)


# A function to preprocess user inputs to match the model's training data format
def create_input_dataframe(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    """
    Preprocesses the raw user inputs to match the feature names and format
    expected by the new trained model's pipeline.
    
    This function handles the calculation of BMI and the creation of
    new categorical features for BMI and blood pressure.
    """
    
    # Calculate BMI
    height_m = height / 100
    bmi = weight / (height_m**2) if height_m > 0 else 0
    
    # Determine BMI category
    bmi_category = ""
    if bmi < 25:
        bmi_category = "Normal"
    elif 25 <= bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    # Determine Blood Pressure category
    bp_category = ""
    if ap_hi < 130 and ap_lo < 85:
        bp_category = "Normal"
    elif 130 <= ap_hi < 140 or 85 <= ap_lo < 90:
        bp_category = "Hypertension Stage 1"
    else:
        bp_category = "Hypertension Stage 2"

    # The columns the model was trained on
    feature_columns = [
        'age', 'height', 'weight', 'ap_hi', 'ap_lo',
        'age_years', 'bmi',
        'gender_2',
        'cholesterol_2', 'cholesterol_3',
        'gluc_2', 'gluc_3',
        'smoke_1',
        'alco_1',
        'active_1',
        'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese',
        'bp_category_Hypertension Stage 1', 'bp_category_Hypertension Stage 2', 'bp_category_Normal',
        'age_group_40-49', 'age_group_50-59', 'age_group_60-65'
    ]
    
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    input_df['age'] = age
    input_df['age_years'] = age
    input_df['height'] = height
    input_df['weight'] = weight
    input_df['ap_hi'] = ap_hi
    input_df['ap_lo'] = ap_lo
    input_df['bmi'] = bmi
    
    input_df[f'gender_{gender}'] = 1
    if cholesterol > 1:
        input_df[f'cholesterol_{cholesterol}'] = 1
    if gluc > 1:
        input_df[f'gluc_{gluc}'] = 1
    if smoke == 1:
        input_df['smoke_1'] = 1
    if alco == 1:
        input_df['alco_1'] = 1
    if active == 1:
        input_df['active_1'] = 1
    
    input_df[f'bmi_category_{bmi_category}'] = 1
    input_df[f'bp_category_{bp_category}'] = 1
    
    if 40 <= age <= 49:
        input_df['age_group_40-49'] = 1
    elif 50 <= age <= 59:
        input_df['age_group_50-59'] = 1
    elif 60 <= age <= 65:
        input_df['age_group_60-65'] = 1
    
    return input_df

# --- Sidebar for additional info ---
with st.sidebar:
    st.header("About this App")
    st.markdown("""
        This application uses a Gradient Boosting Classifier, trained on a large dataset,
        to predict the risk of cardiovascular disease. The model's pipeline includes
        several preprocessing steps, such as feature scaling, BMI calculation, and
        blood pressure categorization, to ensure accurate predictions.
        
        Use the form to the right to input a patient's health data and
        get an immediate risk assessment.
    """)
    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit")

# --- Main App Content ---
st.title("❤️ Advanced Cardiovascular Disease Predictor")
st.markdown("""
    Please enter the patient's health details below. The model will analyze the data
    and provide a risk assessment for cardiovascular disease.
""")

# Use columns and expanders to create a cleaner layout
col1, col2 = st.columns(2)

with col1:
    with st.expander("👤 Patient Information", expanded=True):
        age = st.number_input("Age (in years)", min_value=1, max_value=100, value=50, help="Age of the patient in years.")
        gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
        height = st.number_input("Height (in cm)", min_value=50, max_value=250, value=170)
        weight = st.number_input("Weight (in kg)", min_value=10, max_value=250, value=75)
    
    with st.expander("🏥 Medical Metrics", expanded=True):
        ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=70, max_value=250, value=120, help="The top number in a blood pressure reading.")
        ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=40, max_value=150, value=80, help="The bottom number in a blood pressure reading.")
        cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
        gluc = st.selectbox("Glucose Level", options=[1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])

with col2:
    with st.expander("💪 Lifestyle Factors", expanded=True):
        smoke = st.selectbox("Smoker?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        alco = st.selectbox("Alcohol Intake?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        active = st.selectbox("Physically Active?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    # Display calculated BMI as a metric for user feedback
    height_m = height / 100
    bmi = weight / (height_m**2) if height_m > 0 else 0
    st.metric("Body Mass Index (BMI)", f"{bmi:.2f}")

# --- Prediction button and logic ---
st.markdown("---")
if st.button("Predict Cardiovascular Disease", use_container_width=True, type="primary"):
    # Preprocess the user input before passing it to the model
    input_data = create_input_dataframe(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active)

    # Make the prediction
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        st.subheader("Prediction Result")
        
        # Display the result with a custom styled box and progress bar
        if prediction[0] == 1:
            st.markdown(f'<div class="prediction-box prediction-high"><h3>High Risk of Cardiovascular Disease</h3></div>', unsafe_allow_html=True)
            st.progress(prediction_proba[0][1], text=f"High Risk: {prediction_proba[0][1]:.2f}")
        else:
            st.markdown(f'<div class="prediction-box prediction-low"><h3>Low Risk of Cardiovascular Disease</h3></div>', unsafe_allow_html=True)
            st.progress(prediction_proba[0][0], text=f"Low Risk: {prediction_proba[0][0]:.2f}")

        st.info("""
            **Disclaimer:** This is a machine learning prediction and not a medical diagnosis.
            Consult a healthcare professional for accurate medical advice.
        """)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
