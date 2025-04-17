import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Load the trained model
model = tf.keras.models.load_model('food_detection_model.h5')

# Subcategory labels (these are your actual model output classes)
subcategories = [
    'Apple', 'Banana', 'Burger', 'Coffee', 'Fries', 'Noodles', 'Onion', 'Peas', 'Pizza', 'Potato', 'Soft Drink', 'Tea', 'kiwi'
]

# Streamlit page configuration
st.set_page_config(page_title="Food Classifier", layout="centered")

# Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("🍔 Food Classifier")
st.write("Upload an image of a food item, and we'll tell you what it is!")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Preprocess image for MobileNetV2
    # Preprocess image for MobileNetV2
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # <-- crucial
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    st.write("🔍 Classifying...")
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_subcategory = subcategories[predicted_index]
    confidence = np.max(predictions) * 100

    # Show result
    st.success(f"🍱 Predicted Food Item: **{predicted_subcategory.title()}**")
    st.info(f"📈 Confidence: **{confidence:.2f}%**")
