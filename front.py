import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your MobileNetV2 model
#model = tf.keras.models.load_model('mobilenetv2_food_classifier.h5')

# Subcategories and mapping to categories
subcategories = [
    'tea', 'coffee', 'soft drinks',
    'apple', 'banana', 'kiwi',
    'onion', 'potato', 'peas',
    'pizza', 'burger', 'fries', 'noodles'
]

subcategory_to_category = {
    'tea': 'Beverages',
    'coffee': 'Beverages',
    'soft drinks': 'Beverages',
    'apple': 'Fruits',
    'banana': 'Fruits',
    'kiwi': 'Fruits',
    'onion': 'Veggies',
    'potato': 'Veggies',
    'peas': 'Veggies',
    'pizza': 'Fast Food',
    'burger': 'Fast Food',
    'fries': 'Fast Food',
    'noodles': 'Fast Food'
}

# Streamlit Page Config
st.set_page_config(page_title="Food Classifier", layout="centered")
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

st.title("🍔 Food Classifier")
st.write("Upload an image of a food item and we'll tell you what it is!")

# File Uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess for MobileNetV2
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    st.write("🔍 Classifying...")
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_subcategory = subcategories[predicted_index]
    predicted_category = subcategory_to_category[predicted_subcategory]
    confidence = np.max(predictions) * 100

    # Output
    st.success(f"🍱 Predicted Subcategory: **{predicted_subcategory.title()}**")
    st.info(f"🗂️ Category: **{predicted_category}**")
    st.info(f"🔎 Confidence: {confidence:.2f}%")
