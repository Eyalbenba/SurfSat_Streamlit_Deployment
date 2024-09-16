import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import pickle
import numpy as np
import os
import random

# === Hardcoded variables ===
# Path to the saved model
MODEL_PATH = r'trained_model.pkl'

# Path to the default image displayed at the top
DEFAULT_IMAGE_PATH = r'data/Streamlit_open_image.jpg'

# Path to the directory containing sample images
SAMPLE_IMAGE_DIR = r'data/Photos Try'  


# === Functions ===

# Load the model from the .pkl file
def load_model_from_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)
    model.eval()  # Set the model to evaluation mode
    return model


# Load the model
model = load_model_from_pickle(MODEL_PATH)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define image transformation
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Inject custom CSS for button styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #1C86EE;
    }
    </style>
    """, unsafe_allow_html=True)

# === Streamlit App ===

# Display the default image at the top of the app
st.image(DEFAULT_IMAGE_PATH, use_column_width=True)

# Streamlit app title
st.title("SurfSat: AI Surf Spot Classifier")

# Add the link to your Medium article
st.markdown("""
For Full Code and Story: [Medium Article](https://medium.com/p/4fe00251f930)
""")

# Add instructions for the user
st.markdown("""
Please upload a **satellite image** with at least **224x224** resolution.
""")

# Initialize session state variables if not already present
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if 'prediction' not in st.session_state:
    st.session_state.prediction = None

if 'sample_image' not in st.session_state:
    st.session_state.sample_image = None

if 'reset' not in st.session_state:
    st.session_state.reset = False


# Function to randomly select an image from a directory
def select_random_image_from_dir(directory):
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_image = random.choice(image_files)
    return os.path.join(directory, selected_image)


# Function to make a prediction on the selected image
def make_prediction(image):
    # Transform the image
    input_image = transforms_test(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make prediction
    with torch.no_grad():
        output = model(input_image)
        _, predicted = torch.max(output, 1)

    # Interpret prediction
    class_names = ['Potentially Good for Surf', 'Probably Not Good for Surf']
    return class_names[predicted.item()]


# Function to handle image upload and prediction
def initiate_session():
    # Upload the image
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Store the uploaded file in session state
        st.session_state.uploaded_file = uploaded_file

    # Proceed if there is an uploaded file
    if st.session_state.uploaded_file is not None:
        # Load the image from session state
        image = Image.open(st.session_state.uploaded_file)

        # Convert image to RGB if it's not
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Check image size
        if image.size[0] < 224 or image.size[1] < 224:
            st.error("Image must be at least 224x224 pixels. Please upload a larger image.")
            return

        # Show the image in the app
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        st.session_state.prediction = make_prediction(image)

        # Display the prediction
        display_prediction()


# Function to handle sample image selection and prediction
def use_sample_image():
    selected_image_path = select_random_image_from_dir(SAMPLE_IMAGE_DIR)
    image = Image.open(selected_image_path)

    # Convert image to RGB if it's not
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Show the sample image in the app
    st.image(image, caption='Selected Sample Image', use_column_width=True)

    # Make prediction
    st.session_state.prediction = make_prediction(image)

    # Store the sample image in session state
    st.session_state.sample_image = selected_image_path

    # Display the prediction
    display_prediction()


# Function to display the prediction result in a more appealing way
def display_prediction():
    if st.session_state.prediction is not None:
        if st.session_state.prediction == 'Potentially Good for Surf':
            st.markdown(
                f"""
                <div style="background-color:#d4edda;padding:10px;border-radius:10px;">
                    <h2 style="color:#155724;"><strong> Prediction: {st.session_state.prediction} 🌊</strong></h2>
                    <p style="color:#155724;">Looks like a great spot to catch some waves! 🏄‍♂️</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background-color:#f8d7da;padding:10px;border-radius:10px;">
                    <h2 style="color:#721c24;"><strong> Prediction: {st.session_state.prediction}🌧️</strong></h2>
                    <p style="color:#721c24;">Unfortunately, it may not be the best beach for surfing. 😔</p>
                </div>
                """,
                unsafe_allow_html=True
            )


# Always display the upload widget first
initiate_session()

# Add the "Try One of Our Photos" button below the uploader
if st.button('Try One of Our Photos'):
    use_sample_image()
