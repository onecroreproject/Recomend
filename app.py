import streamlit as st
import sqlite3
import hashlib
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import joblib

loaded_model = joblib.load("xgboost_diet_recommendation.pkl")


def get_user_input(columns):
    user_input = []
    for column in columns:
        value = st.number_input(f"Enter value for {column}:", value=0.0)  # Get numeric input
        user_input.append(value)
    return [user_input]  # Reshape to match model input format



# Function to set background for a specific page (with a custom image)
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

# Function to create a connection to SQLite database
def create_connection():
    conn = sqlite3.connect('users.db')  # Connect to or create the database file
    return conn

# Function to create users table if it doesn't exist
def create_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

# Function to hash the password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to check if the user exists in the database
def user_exists(username):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

# Function to register the user in the database
def register_user(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                       (username, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

# Function to validate login credentials
def validate_login(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", 
                   (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    return user is not None

# Login page function
def login_page():
    set_background_image("https://img.freepik.com/free-photo/design-space-paper-textured-background_53876-14888.jpg?semt=ais_hybrid")
    
    st.title("Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if validate_login(username, password):
            st.success("Login Successful!")
            st.session_state.logged_in = True
            st.session_state.username = username
            # Redirect to the next page after successful login
            st.rerun()

        else:
            st.error("Invalid username or password.")
            


# Register page function
def register_page():
    set_background_image("https://img.freepik.com/free-photo/design-space-paper-textured-background_53876-14888.jpg?semt=ais_hybrid")
    
    st.title("REGISTER")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if password == confirm_password:
            if register_user(username, password):
                st.success("Registration Successful! Please log in.")
                st.session_state.page = "login"
                st.rerun()

            else:
                st.error("Username already exists.")
        else:
            st.error("Passwords do not match.")



def page3_diet_recommendation():
    set_background_image("https://img.freepik.com/premium-vector/green-fruit-wallpaper-premium-vector_617762-60.jpg?semt=ais_hybrid")
    
    st.header("Diet Recommendation")
    st.write("Enter your details below to receive a personalized diet recommendation.")
    columns = ['Age', 'Gender', 'Weight_kg', 'Height_cm', 'BMI', 'Disease_Type', 'Severity',
           'Physical_Activity_Level', 'Daily_Caloric_Intake', 'Cholesterol_mg/dL', 
           'Blood_Pressure_mmHg', 'Glucose_mg/dL', 'Dietary_Restrictions', 'Allergies', 
           'Preferred_Cuisine', 'Weekly_Exercise_Hours', 'Adherence_to_Diet_Plan', 
           'Dietary_Nutrient_Imbalance_Score']
    # Get user input
    user_data = get_user_input(columns)

    # If the user has filled in the data and presses the button
    if st.button("Get Diet Recommendation"):
        if any(value == 0 for value in user_data[0]):
            st.error("Please enter all the required values!")
        else:
            # Predict using the trained model
            prediction = loaded_model.predict(user_data)[0]  # Get single prediction

            # Mapping prediction to diet type
            diet_mapping = {0: "Balanced", 1: "Low_Carb", 2: "Low_Sodium"}  # Update based on label encoding
            predicted_diet = diet_mapping.get(prediction, "Unknown")

            # Print prediction
            st.subheader(f"Predicted Diet Recommendation: {predicted_diet}")

            # Provide Recommendations Based on Prediction
            if predicted_diet == "Balanced":
                st.write("Recommendation: Maintain a well-balanced diet including proteins, carbs, and healthy fats.")
            elif predicted_diet == "Low_Carb":
                st.write("Recommendation: Reduce carb intake. Focus on proteins, healthy fats, and fiber-rich foods.")
            elif predicted_diet == "Low_Sodium":
                st.write("Recommendation: Cut down on salt. Increase potassium-rich foods like bananas and leafy greens.")
            else:
                st.write("Recommendation: No specific recommendation available.")






# Hair type prediction page function
def page2_hair_type_prediction():
    set_background_image("https://img.freepik.com/free-photo/professional-hair-supplies-flat-lay_23-2148352852.jpg?semt=ais_hybrid")

    st.header("Hair Type Prediction")
    st.write("Upload an image of your hair to predict its type and get personalized food recommendations.")

    # Predefined labels and food recommendations
    class_labels = ["Type 1 (Straight)", "Type 2 (Wavy)", "Type 3 (Curly)"]

    food_recommendations = {
        "Type 1 (Straight)": "Eat lean protein (chicken, eggs, fish), omega-3s (salmon, walnuts), and vitamin-rich foods (carrots, citrus, spinach).",
        "Type 2 (Wavy)": "Include biotin-rich foods (avocado, almonds, eggs), zinc & iron (pumpkin seeds, lentils), and hydrating foods (berries, cucumber).",
        "Type 3 (Curly)": "Consume healthy fats (olive oil, avocado, fatty fish), collagen-boosting foods (oranges, berries, bone broth), and magnesium-rich foods (Brazil nuts, leafy greens)."
    }

    def predict_hair_type(image_path):
        loaded_model = load_model("hair_type_classifier.keras")
        
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        prediction = loaded_model.predict(img_array)
        predicted_class = np.argmax(prediction)
        
        hair_type = class_labels[predicted_class]
        food_tips = food_recommendations.get(hair_type, "No specific recommendation available.")
        
        return hair_type, food_tips

    # File uploader for the user to upload their hair image
    image_file = st.file_uploader("Upload your hair image", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        # Display the image
        st.image(image_file, caption="Uploaded Image", use_column_width=True)
        
        # Save the uploaded image to a temporary file for prediction
        with open("temp_image.jpg", "wb") as f:
            f.write(image_file.getbuffer())
        
        # Predict hair type and food recommendations
        hair_type, food_tips = predict_hair_type("temp_image.jpg")

        # Display the results
        st.subheader(f"Predicted Hair Type: {hair_type}")
        st.subheader(f"Food Recommendation: {food_tips}")





def predict_skin_condition(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    loaded_model1 = tf.keras.models.load_model("skin_condition_classifier.keras")
    prediction = loaded_model1.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_labels1 = ["Acne", "Carcinoma", "Eczema", "Keratosis", "Milia", "Rosacea"]
    
    skin_condition = class_labels1[predicted_class]

    # ✅ Provide Skincare & Food Recommendations
    recommendations = {
        "Acne": {
            "Skincare": "Use salicylic acid or benzoyl peroxide. Avoid oily foods and drink plenty of water.",
            "Food": "Eat omega-3-rich foods (salmon, walnuts), zinc sources (pumpkin seeds), and probiotics (yogurt). Avoid dairy and sugar."
        },
        "Carcinoma": {
            "Skincare": "Consult a dermatologist immediately. Avoid direct sunlight and use SPF 50+ sunscreen.",
            "Food": "Increase antioxidants (berries, green tea), leafy greens (spinach), and vitamin A-rich foods (carrots, sweet potatoes)."
        },
        "Eczema": {
            "Skincare": "Use gentle moisturizers. Avoid harsh soaps and keep skin hydrated.",
            "Food": "Consume anti-inflammatory foods (turmeric, ginger), vitamin E-rich foods (almonds, avocados), and omega-3 sources."
        },
        "Keratosis": {
            "Skincare": "Exfoliate regularly and consult a skin specialist.",
            "Food": "Include vitamin D sources (mushrooms), healthy fats (olive oil, nuts), and protein-rich foods. Reduce sugar."
        },
        "Milia": {
            "Skincare": "Use gentle exfoliation. Avoid thick creams and opt for non-comedogenic products.",
            "Food": "Eat vitamin A-rich foods (carrots, sweet potatoes), hydration-boosting foods (cucumber, watermelon), and lean proteins."
        },
        "Rosacea": {
            "Skincare": "Avoid spicy foods and alcohol. Use gentle skincare and protect skin from extreme temperatures.",
            "Food": "Consume anti-inflammatory foods (green tea, blueberries), cooling foods (cucumber, celery), and vitamin C-rich foods."
        }
    }
    
    rec = recommendations.get(skin_condition, {"Skincare": "No specific recommendation available.", "Food": "No food recommendation available."})
    
    return skin_condition, rec["Skincare"], rec["Food"]




def page4_skin_condition_prediction():
    set_background_image("https://img.freepik.com/free-photo/spa-spoon-soap-copy-space_23-2148481761.jpg?semt=ais_hybrid")
    
    st.header("Skin Condition Prediction")
    st.write("Upload an image of your skin to predict the condition and receive recommendations.")

    # Image upload
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Save the uploaded image temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        # Show the uploaded image to the user
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Predict skin condition
        skin_condition, skincare_rec, food_rec = predict_skin_condition("temp_image.jpg")
        
        # Display results
        st.subheader(f"Predicted Skin Condition: {skin_condition}")
        st.write(f"**Skincare Recommendation:** {skincare_rec}")
        st.write(f"**Food Recommendation:** {food_rec}")




def main():
    create_table()  # Ensure that the users table is created

    # Check if the user is logged in or not
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        # If not logged in, show the login page
        page = st.sidebar.selectbox("Select a page", [
            "Login", 
            "Register", 
        ])
        if page == "Login":
            login_page()
        else:
            register_page()
    else:
        # Display the sidebar for page navigation after login
        page = st.sidebar.selectbox("Select a page", [
            "Food Recognition", 
            "Hair Type Classification", 
            "Skin Care Recommendation"
        ])
        
        # Add logout button
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False  # Clear logged_in session state
            st.session_state.username = ""  # Optionally clear username
            st.success("Logout Successful!")
            st.session_state.page = "login"  # Optionally redirect to login page
            st.rerun()  # Rerun to refresh the page and show login screen
        
        # Handle navigation based on selected page
        if page == "Food Recognition":
            page3_diet_recommendation()  # Show food recognition page
        elif page == "Hair Type Classification":
            page2_hair_type_prediction()  # Show hair type prediction page
        elif page == "Skin Care Recommendation":
            page4_skin_condition_prediction()  # Show skin care recommendation page


# Run the app
if __name__ == "__main__":
    main()
