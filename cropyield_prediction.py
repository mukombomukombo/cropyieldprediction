import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load the pre-trained model (XGBoost pipeline)
@st.cache_resource
def load_model():
    try:
        # Add error handling with version info
        model = joblib.load('cropyield.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the model was trained with scikit-learn 1.6.1")
        return None

# Load the model
model = load_model()

# Title of the web app
st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")
st.title("🌾 Crop Yield Prediction App")
st.markdown("Predict crop yield based on environmental and agricultural factors")

# Sidebar for input fields
st.sidebar.header("📊 Input Parameters")
st.sidebar.markdown("Enter the following details to predict crop yield:")

# Input fields
area = st.sidebar.selectbox(
    'Area/Country',
    options=['Zambia', 'Zimbabwe', 'South Africa', 'Kenya', 'Nigeria', 'Egypt', 'India', 'China', 'Brazil', 'USA', 'Germany', 'France', 'Australia']
)

item = st.sidebar.selectbox(
    'Crop Type (Item)',
    options=['Maize', 'Wheat', 'Rice', 'Soybeans', 'Potatoes', 'Cassava', 'Barley', 'Sorghum', 'Millet', 'Sugar cane']
)

year = st.sidebar.number_input('Year', min_value=1990, max_value=2030, value=2024, step=1)

rainfall = st.sidebar.number_input(
    'Average Rainfall (mm/year)', 
    min_value=0.0, 
    max_value=5000.0, 
    value=950.0, 
    step=10.0,
    help="Average annual rainfall in millimeters"
)

pesticides = st.sidebar.number_input(
    'Pesticides Used (tonnes)', 
    min_value=0.0, 
    max_value=100000.0, 
    value=25.0, 
    step=5.0,
    help="Total pesticides used in tonnes"
)

temperature = st.sidebar.number_input(
    'Average Temperature (°C)', 
    min_value=-10.0, 
    max_value=50.0, 
    value=24.5, 
    step=0.5,
    help="Average annual temperature in degrees Celsius"
)

# Button to trigger prediction
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button('🔮 Predict Crop Yield', use_container_width=True)

if predict_button:
    if model is None:
        st.error("Model not loaded. Please check the model file and scikit-learn version.")
    else:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Area': [area],
            'Item': [item],
            'Year': [year],
            'rainfall': [rainfall],
            'pesticides': [pesticides],
            'temperature': [temperature]
        })
        
        # Make prediction
        try:
            # Predict using the pipeline (log-transformed yield)
            log_prediction = model.predict(input_data)
            
            # Convert back to original scale (hg/ha)
            prediction_hg_ha = np.expm1(log_prediction)[0]
            
            # Convert to kg/ha
            prediction_kg_ha = prediction_hg_ha / 100
            
            # Convert to tons per hectare
            prediction_tons_ha = prediction_kg_ha / 1000
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            # Create metrics display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="🌽 Yield (hg/ha)", 
                    value=f"{prediction_hg_ha:,.0f}",
                    help="Hectograms per hectare (1 hg = 100 grams)"
                )
            
            with col2:
                st.metric(
                    label="📦 Yield (kg/ha)", 
                    value=f"{prediction_kg_ha:,.0f}",
                    help="Kilograms per hectare"
                )
            
            with col3:
                st.metric(
                    label="🚜 Yield (tons/ha)", 
                    value=f"{prediction_tons_ha:,.2f}",
                    help="Tons per hectare"
                )
            
            # Additional information
            st.markdown("---")
            st.subheader("📋 Prediction Details")
            
            # Create a dataframe with input parameters
            input_summary = pd.DataFrame({
                'Parameter': ['Area', 'Crop Type', 'Year', 'Rainfall (mm/year)', 'Pesticides (tonnes)', 'Temperature (°C)'],
                'Value': [area, item, year, f"{rainfall:,.0f}", f"{pesticides:,.0f}", f"{temperature:.1f}"]
            })
            st.table(input_summary)
            
            # Add interpretation
            st.markdown("---")
            st.subheader("💡 Interpretation")
            
            if prediction_tons_ha < 2:
                st.warning(f"⚠️ Low yield predicted ({prediction_tons_ha:.2f} tons/ha). Consider improving irrigation, pest control, or soil management.")
            elif prediction_tons_ha < 5:
                st.info(f"📈 Moderate yield predicted ({prediction_tons_ha:.2f} tons/ha). Good potential with room for improvement.")
            else:
                st.success(f"🎉 High yield predicted ({prediction_tons_ha:.2f} tons/ha)! Excellent conditions for this crop.")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please check if all inputs are valid and try again.")

# Add footer information

