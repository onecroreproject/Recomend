import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Set background for a specific page (with a custom image)
def set_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('{image_url}');
            background-size: cover;
            background-position: center;
            height: 100vh;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Login page
def login_page():
    set_background_image("https://media.ahmedabadmirror.com/am/uploads/mediaGallery/image/1722965065995.jpg-org")  # Custom background for the login page
    
    st.title("WELCOME TO BEYOUTIFAI")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.success("Login Successful!")
            st.session_state.logged_in = True
            # Redirect to the next page after successful login
            st.experimental_rerun()
        else:
            st.error("Invalid username or password.")

# Food Recognition (Page 1)
@st.cache_resource
def load_pretrained_model():
    model = MobileNetV2(weights="imagenet")
    return model

model = load_pretrained_model()

def page1_food_recognition():
    set_background_image("https://img.freepik.com/free-photo/bread-slices-with-topping-tomato-cheese-olives-white-table_23-2148194999.jpg?semt=ais_hybrid")  # Custom background for Food Recognition page
    st.title("Food Recognition & Calorie Estimation")
    st.write("Upload an image of your food, and we'll recognize it and estimate its calorie content!")
    
    calorie_database = {
        "apple": 52, 
        "banana": 96,
        "burger": 295,
        "pizza": 266,
        "salad": 33,
        "sushi": 200,
    }

    uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_size = (224, 224)
        image_resized = image.resize(img_size)
        image_array = np.array(image_resized)
        image_preprocessed = preprocess_input(image_array)
        image_preprocessed = np.expand_dims(image_preprocessed, axis=0)

        predictions = model.predict(image_preprocessed)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        st.write("### Predictions:")
        for i, (imagenet_id, label, confidence) in enumerate(decoded_predictions):
            st.write(f"{i+1}. **{label.capitalize()}**: {confidence * 100:.2f}%")

        recognized_foods = [label.lower() for _, label, _ in decoded_predictions]
        for food in recognized_foods:
            if food in calorie_database:
                st.write(f"### Recognized Food: {food.capitalize()}")
                st.write(f"Estimated Calories: {calorie_database[food]} kcal per 100g")
                break
        else:
            st.write("Sorry, the food item is not in our calorie database.")

# Personalized Diet Suggestion (Page 2)
def page2_diet_exercise_suggestion():
    set_background_image("https://img.freepik.com/premium-photo/healthy-food-variation-with-copy-space-empty-background_944892-519.jpg")  # Custom background for Diet Suggestion page
    st.title("Personalized Diet Suggestion")
    st.write("Enter your details to get personalized diet suggestions.")

    # Input fields
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    height = st.number_input("Height (in cm)", min_value=50, max_value=250, value=170)
    weight = st.number_input("Weight (in kg)", min_value=10, max_value=300, value=70)
    food_preference = st.selectbox("Food Preference", ["Vegetarian", "Non-Vegetarian"])

    # Diet suggestion based on weight
    def get_diet_by_weight(weight):
        if weight <= 50:
            return ["Increase protein intake", "Include high-calorie snacks"]
        elif 50 < weight <= 70:
            return ["Balanced diet", "Include more vegetables and lean proteins"]
        elif 70 < weight <= 100:
            return ["Focus on weight management", "Avoid high-carb foods"]
        else:
            return ["Reduce calorie intake", "Include more fruits and vegetables"]

    # Diet suggestion based on height
    def get_diet_by_height(height):
        if height <= 160:
            return ["Smaller portions, focus on protein intake", "Avoid processed snacks"]
        elif 160 < height <= 180:
            return ["Moderate portions, balanced diet", "Increase fiber intake"]
        else:
            return ["Higher calorie intake for energy", "Include complex carbs"]

    if st.button("Generate Diet"):
        # Get diet suggestions
        weight_diet = get_diet_by_weight(weight)
        height_diet = get_diet_by_height(height)

        st.write("### Suggested Diet Plan based on Weight and Height:")
        st.write("**Weight-based Suggestions:**")
        for item in weight_diet:
            st.write(f"- {item}")

        st.write("**Height-based Suggestions:**")
        for item in height_diet:
            st.write(f"- {item}")

        # Optional: Add the BMI suggestion
        bmi = weight / ((height / 100) ** 2)
        st.write(f"### BMI: {bmi:.2f}")
        if bmi < 18.5:
            st.write("You are underweight. Consider increasing calorie intake.")
        elif 18.5 <= bmi <= 24.9:
            st.write("You have a healthy weight. Maintain a balanced diet.")
        elif 25 <= bmi <= 29.9:
            st.write("You are overweight. Focus on weight management.")
        else:
            st.write("You are obese. Reduce calorie intake and consider consulting a healthcare provider.")

# Target Weight-Based Diet and Exercise Plan (Page 3)
def page3_target_weight_based_plan():
    set_background_image("https://img.freepik.com/premium-photo/healthy-food-ingredients-blue-background_1249787-40657.jpg")  # Custom background for Target Weight page
    st.title("Target Weight-Based Diet and Exercise Plan")
    st.write("Enter your current and target weight to get personalized diet and exercise recommendations.")

    current_weight = st.number_input("Current Weight (kg)", min_value=10, max_value=300, value=70)
    target_weight = st.number_input("Target Weight (kg)", min_value=10, max_value=300, value=60)
    age = st.number_input("Age", min_value=10, max_value=90, value=25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    if st.button("Generate Plan"):
        weight_change = current_weight - target_weight
        st.write(f"### Weight Change Goal: {weight_change} kg")

        if weight_change > 0:
            st.write("### Suggested Diet Plan for Weight Loss:")
            if 10 <= age <= 25:
                st.write("1. Focus on nutrient-dense, low-calorie meals.")
                st.write("2. Include lean protein sources, avoid sugary drinks.")
            elif 25 < age <= 45:
                st.write("1. Opt for high-fiber foods.")
                st.write("2. Avoid refined carbs.")
            else:
                st.write("1. Incorporate whole grains and leafy greens.")
                st.write("2. Reduce salt and fat intake.")

            st.write("### Suggested Exercise Plan for Weight Loss:")
            if 10 <= age <= 25:
                st.write("1. High-intensity interval training (HIIT).")
                st.write("2. Daily 30-minute brisk walks.")
            elif 25 < age <= 45:
                st.write("1. 45-minute cardio sessions, 4 times a week.")
                st.write("2. Strength training twice a week.")
            else:
                st.write("1. Light aerobics and yoga.")
                st.write("2. Gentle stretching routines.")

        else:
            st.write("### Suggested Diet Plan for Weight Gain:")
            if 10 <= age <= 25:
                st.write("1. Increase calorie intake with nuts and dairy.")
                st.write("2. Eat frequent small meals.")
            elif 25 < age <= 45:
                st.write("1. Focus on healthy fats and whole grains.")
                st.write("2. Include protein-rich snacks.")
            else:
                st.write("1. Soft, nutrient-dense foods.")
                st.write("2. Ensure adequate protein intake.")

            st.write("### Suggested Exercise Plan for Weight Gain:")
            if 10 <= age <= 25:
                st.write("1. Weight lifting 4 times a week.")
                st.write("2. Include compound movements like squats and deadlifts.")
            elif 25 < age <= 45:
                st.write("1. Strength training thrice a week.")
                st.write("2. Moderate-intensity cardio.")
            else:
                st.write("1. Gentle strength exercises.")
                st.write("2. Light physical activities to maintain mobility.")

        st.write("### Age-Based Recommendations:")
        if 10 <= age <= 15:
            st.write("Focus on proper growth with a balanced diet and moderate physical activities.")
        elif 15 < age <= 25:
            st.write("Include strength-building exercises and a high-protein diet.")
        elif 25 < age <= 35:
            st.write("Maintain a balanced diet and incorporate regular cardio.")
        elif 35 < age <= 45:
            st.write("Focus on stress management and maintain muscle mass.")
        elif 45 < age <= 60:
            st.write("Include joint-friendly exercises and monitor cholesterol levels.")

# Machine Learning Performance Evaluation (Page 4)
def page4_ml_analysis():
    set_background_image("https://t4.ftcdn.net/jpg/06/33/27/09/360_F_633270910_NcK1JCWzqyul9b189PKabKp2o4PqpVKX.jpg")  # Custom background for Machine Learning page
    st.title("Machine Learning Model Performance Analysis")
    st.write("Here are the performance metrics of the food recognition model:")

    # Assuming you have a model's predictions and ground truth for evaluation
    true_labels = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0]  # Actual ground truth
    predicted_labels = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0]  # Predicted labels by model

    # Calculate performance metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    st.write(f"### Accuracy: {accuracy * 100:.2f}%")
    st.write(f"### Precision: {precision * 100:.2f}%")
    st.write(f"### Recall: {recall * 100:.2f}%")
    st.write(f"### F1 Score: {f1 * 100:.2f}%")

    # Plotting the metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1]

    fig, ax = plt.subplots()
    ax.bar(metrics, values, color=['green', 'blue', 'orange', 'red'])
    ax.set_ylabel("Percentage")
    ax.set_title("Model Performance Metrics")
    st.pyplot(fig)


# Skin Care Recommendation (Page 5)
def page5_skin_care():
    set_background_image("https://img.freepik.com/premium-photo/healthy-food-ingredients-blue-background_1249787-40657.jpg")  # Custom background for Skin Care page
    st.title("Skin Care Recommendation")
    st.write("Enter your current skin image to get personalized recommendations.")

    def get_recommendation(prediction_type):
        if prediction_type == "Stress-Induced Hairloss":
            return """
            **Recommendation:**
            - **Stress management:** Practice relaxation techniques like meditation, yoga, or deep breathing exercises.
            - **Exercise regularly** to reduce stress levels and improve blood circulation.
            - **Consider therapy or counseling** if stress is affecting your overall well-being.
            - **Adequate sleep:** Try to get 7-8 hours of sleep every night to support hair growth.
            - **Avoid excessive caffeine** and other stimulants that may worsen stress.
            """
        
        elif prediction_type == "Nutritional Deficiency":
            return """
            **Recommendation:**
            - **Improve your diet:** Include more fruits, vegetables, and proteins like eggs, nuts, and seeds in your meals.
            - **Vitamins & Supplements:** Consider taking Biotin, Vitamin C, and Omega-3 fatty acids, which support hair health.
            - **Increase iron intake:** Include iron-rich foods like spinach, lentils, and red meat in your diet.
            - **Hydration:** Drink plenty of water to keep your scalp hydrated and promote hair growth.
            - **Consult a nutritionist** for a personalized diet plan to ensure you’re meeting all your nutritional needs.
            """
        
        elif prediction_type == "Allergic Reaction (Product-related)":
            return """
            **Recommendation:**
            - **Discontinue use of new products** and consult a dermatologist to identify the allergic ingredient.
            - **Patch test new products** before using them to check for any allergic reactions.
            - **Soothing oils:** Use natural oils like coconut oil, aloe vera, or argan oil to calm scalp irritation.
            - **Choose hypoallergenic products** that are less likely to cause irritation.
            - **Avoid harsh treatments** (like coloring or chemical hair straightening) until your scalp heals.
            """
        
        elif prediction_type == "Post-Medication Hairloss":
            return """
            **Recommendation:**
            - **Consult your doctor**: If hair fall began after taking antibiotics or other medications, speak with your doctor about possible alternatives.
            - **Use probiotics** to restore gut health, which can support overall well-being and hair growth.
            - **Avoid harsh treatments** and let your hair recover naturally.
            - **Increase intake of hair-supporting nutrients** like Vitamin D, Vitamin E, and Biotin.
            - **Monitor for any side effects** from your medication and report them to your healthcare provider.
            """
        
        elif prediction_type == "Chronic Hairloss":
            return """
            **Recommendation:**
            - **Consult a dermatologist** to rule out any underlying medical conditions, such as hormonal imbalances or autoimmune diseases.
            - **Consider a scalp treatment plan** that includes oil massages to stimulate hair growth.
            - **Nutrition Check:** Ensure your diet includes enough proteins and essential fatty acids for hair growth.
            - **Hair care:** Use gentle, nourishing products for your hair and avoid excessive heat styling.
            - **Consider treatments** like PRP (Platelet-Rich Plasma) therapy if recommended by a specialist.
            """
        
        else:
            return "No recommendations available for this condition."

    # Streamlit UI
    st.title("Hair Fall Prediction & Recommendation System")

    # User Input Fields
    st.subheader("Step 1: Upload Image (Optional)")
    uploaded_image = st.file_uploader("Upload your hair or skin image", type=["jpg", "jpeg", "png"])

    st.subheader("Step 2: Answer the Questions")

    # Questions for hair fall prediction
    hair_cause = st.selectbox("What could be causing your hair fall?", ["Stress", "Antibiotic use", "Lack of proper food intake"])
    duration = st.selectbox("How long have you been experiencing hair fall?", ["Less than a month", "1-3 months", "More than 3 months"])
    new_products = st.selectbox("Have you recently started using any new hair products?", ["Yes", "No", "I can't remember"])
    symptoms = st.selectbox("Do you notice other symptoms like scalp irritation or dandruff?", ["Yes", "No", "Sometimes"])
    diet = st.selectbox("How would you describe your current diet?", ["Balanced", "Poor in nutrients", "No particular diet"])
    stress = st.selectbox("Are you under a lot of stress or pressure lately?", ["Yes, quite a bit", "A little, but manageable", "No stress at all"])

    # Button to predict
    if st.button("Predict"):
        # Logic to determine the prediction
        if hair_cause == "Stress" and stress == "Yes, quite a bit":
            prediction = "Stress-Induced Hairloss"
        elif diet == "Poor in nutrients":
            prediction = "Nutritional Deficiency"
        elif new_products == "Yes" and symptoms == "Yes":
            prediction = "Allergic Reaction (Product-related)"
        elif duration == "More than 3 months":
            prediction = "Chronic Hairloss"
        elif hair_cause == "Antibiotic use":
            prediction = "Post-Medication Hairloss"
        else:
            prediction = "General Hair Fall"

        # Display prediction and recommendations
        st.subheader(f"Prediction: {prediction}")
        recommendations = get_recommendation(prediction)
        st.markdown(recommendations)

# Modify main to include the Skin Care page
def main():
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        login_page()
    else:
        page = st.sidebar.radio("Select a page", ["Food Recognition", "Diet Suggestion", "Target Weight-Based Plan", "ML Model Analysis", "Skin Care Recommendation"])
        if page == "Food Recognition":
            page1_food_recognition()
        elif page == "Diet Suggestion":
            page2_diet_exercise_suggestion()
        elif page == "Target Weight-Based Plan":
            page3_target_weight_based_plan()
        elif page == "ML Model Analysis":
            page4_ml_analysis()
        elif page == "Skin Care Recommendation":
            page5_skin_care()

if __name__ == "__main__":
    main()

