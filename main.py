import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image

# Path to your model
MODEL_PATH = r"C:/Users/DELL/Desktop/majorproject/Brain Tumor/brainfinal (1).h5"

# Load the model
model = load_model(MODEL_PATH)

# Define the mapping from numerical labels to class names
CLASS_NAMES = {
    0: 'Glioma',
    1: 'Meningioma',
    2: 'No Tumor',
    3: 'Pituitary'
}

def model_predict(image_path, model):
    # Assuming the file is an image, here is a basic preprocessing step
    image = Image.open(image_path)
    image = image.resize((128, 128))  # Resize the image to match the input size expected by your model
    image_array = np.array(image)  # Convert the PIL image to a NumPy array
    image_tensor = image_array.astype('float32') / 255.0  # Normalize the pixel values

    # Add batch dimension
    image_tensor = np.expand_dims(image_tensor, axis=0)

    predictions = model.predict(image_tensor)
    return predictions

# Streamlit app
st.title('Brain Tumor Classification')

st.write("Upload an MRI image to classify the type of brain tumor.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Save the uploaded file temporarily
    temp_file_path = os.path.join('uploads', uploaded_file.name)
    with open(temp_file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Get the prediction
    predictions = model_predict(temp_file_path, model)
    result_index = np.argmax(predictions, axis=1)[0]  # Assuming classification model
    result = CLASS_NAMES.get(result_index, "Unknown")
    st.write(f"Predicted class: {result}")

# Ensure 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')
