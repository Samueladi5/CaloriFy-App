import pandas as pd
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import joblib
import hydralit_components as hc
import hydralit as hy

# Configure Streamlit page
st.set_page_config(
    page_title="CalorieFy",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Synchronize Streamlit states
st.session_state.sync = True
st.session_state.allow_access = True

# Load data and models
@st.cache_resource
def load_resources():
    try:
        abbrev_df = pd.read_csv('usda_branded_food_data.csv')
        processor = AutoImageProcessor.from_pretrained("nateraw/food")
        model = AutoModelForImageClassification.from_pretrained("nateraw/food")
        obesity_model = joblib.load("obesity_model (1).pkl")
        return abbrev_df, processor, model, obesity_model
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None, None, None, None

# Load resources
abbrev_df, processor, food_model, obesity_model = load_resources()

# Initialize HydraApp
app = hy.HydraApp(title='CalorieFy')

# Home Page
@app.addapp(title='HOME', icon="🏠")
def home():
    st.image('Home.png', use_column_width=True)

# Food Classification Page
@app.addapp(title="Count Calories", icon="🍽️")
def food_classification():
    st.header("🍽️ Count Calories")
    image_file = st.file_uploader("Upload an image of your food", type=["jpg", "jpeg", "png"])

    if image_file:
        if processor and food_model:
            with st.spinner("Analyzing image..."):
                try:
                    # Load and process the image
                    image = Image.open(image_file).convert("RGB")
                    inputs = processor(image, return_tensors="pt")
                    outputs = food_model(**inputs)

                    # Get the predicted label
                    predictions = outputs.logits.argmax(-1)
                    food_label = food_model.config.id2label[predictions.item()]

                    # Display predicted food
                    st.success(f"Predicted Food: **{food_label}** 🍴")

                    # Retrieve and display nutritional information
                    if abbrev_df is not None:
                        match = abbrev_df[abbrev_df['name'].str.contains(food_label, case=False, na=False)]
                        if not match.empty:
                            st.write("### Nutritional Information")
                            display_df = match.rename(columns={
                                'Energy': 'Calories',
                                'Protein': 'Protein',
                                'Carbohydrate, by difference': 'Carbohydrates',
                                'Total lipid (fat)': 'Fats'
                            })
                            st.metric("Calories 🔥", f"{display_df['Calories'].iloc[0]} kcal")
                            st.metric("Protein 💪", f"{display_df['Protein'].iloc[0]} g")
                            st.metric("Carbohydrates 🍞", f"{display_df['Carbohydrates'].iloc[0]} g")
                            st.metric("Fats 🥑", f"{display_df['Fats'].iloc[0]} g")
                        else:
                            st.warning("Nutritional information not found for this item.")
                    else:
                        st.warning("Nutritional dataset not loaded.")
                except Exception as e:
                    st.error(f"An error occurred while analyzing the image: {e}")
        else:
            st.warning("Model or processor is not loaded. Please ensure all components are initialized.")
    else:
        st.info("Please upload an image to classify.")

# Obesity Prediction Page
@app.addapp(title="Obesity Prediction", icon="💪")
def obesity_prediction():
    st.header("💪 Obesity Risk Prediction")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.75, step=0.01)
        weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70, step=1)

    with col2:
        family_history = st.selectbox("Family History of Overweight", ["No", "Yes"])
        ncp = st.number_input("Main Meals per Day", min_value=1, max_value=10, value=3)
        caec = st.selectbox("Food between meals?", ["No", "Sometimes", "Frequently", "Always"])
        calc = st.selectbox("Alcohol consumption?", ["No", "Sometimes", "Frequently", "Always"])
        faf = st.number_input("Weekly exercise days", min_value=0, max_value=7, value=2)

    if st.button("Predict Obesity Level") and obesity_model:
        try:
            # Encode inputs for prediction
            input_data = [[age, 1 if gender == "Male" else 0, height, weight,
                           1 if family_history == "Yes" else 0, ncp,
                           {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[caec],
                           {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[calc],
                           faf]]
            # Get prediction
            prediction = obesity_model.predict(input_data)
            predicted_label = prediction[0]

            # Map labels to images and descriptions
            label_to_image = {
                "Normal_Weight": ("normal.jpeg", "Normal Weight"),
                "Overweight_Level_I": ("overweight.jpeg", "Overweight (Type I)"),
                "Overweight_Level_II": ("overweight.jpeg", "Overweight (Type II)"),
                "Overweight_Level_III": ("overweight.jpeg", "Overweight (Type III)"),
                "Obesity_Type_I": ("obese.jpeg", "Obesity (Type I)"),
                "Obesity_Type_II": ("obese.jpeg", "Obesity (Type II)"),
                "Obesity_Type_III": ("obese_besar.jpeg", "Obesity (Type III)")
            }

            # Display result
            if predicted_label in label_to_image:
                image_path, label_description = label_to_image[predicted_label]
                st.image(image_path, caption=label_description, width=100)
                st.success(f"Predicted Obesity Level: {label_description}")
            else:
                st.error("Prediction label is not recognized.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# About Page
@app.addapp(title='About', icon="⚙")
def about():
    st.title("About the Creator")
    col1_width = 200
    col2_width = 800
    col1, col2 = hy.columns([col1_width, col2_width])
    with col1:
        hy.image('boy.png', use_column_width=True)
    with col2:
        hy.markdown("")
        hy.markdown("")
        hy.markdown("")
        hy.markdown("")
        hy.markdown("")
    hy.markdown("""
        **Creator:** 
        Samuel Adi Saut Puryanto  
        
        **Program Studi:** 
        Teknologi Informasi  
        
        **NIM:** 
        21537141018  
        
        **State University of Yogyakarta**
    """)

# Run the app
if __name__ == "__main__":
    app.run()
