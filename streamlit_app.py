import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import pickle
import numpy as np

# Load the model from the .pkl file
def load_model_from_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)
    model.eval()  # Set the model to evaluation mode
    return model

# Path to the saved pickle file
model_path = 'trained_model.pkl'

# Load the model
model = load_model_from_pickle(model_path)

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

# Streamlit app
st.title("Surf Spot Classifier")

# Initialize session state variables if not already present
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if 'prediction' not in st.session_state:
    st.session_state.prediction = None

if 'reset' not in st.session_state:
    st.session_state.reset = False

# Define a callback function to reset the session state and delete the image
def reset_session():
    st.session_state.uploaded_file = None  # Clear the uploaded image
    st.session_state.prediction = None  # Clear the prediction
    st.session_state.reset = True  # Set reset flag to true

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

        # Show the image in the app
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Transform the image
        input_image = transforms_test(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Make prediction
        with torch.no_grad():
            output = model(input_image)
            _, predicted = torch.max(output, 1)

            # Interpret prediction (assuming 0 is 'not good for surf' and 1 is 'good for surf')
            class_names = ['Potentially Good for Surf', 'Probably Not Good for Surf']
            st.session_state.prediction = class_names[predicted.item()]

        # Display the prediction
        st.subheader(f'Prediction: {st.session_state.prediction}')

# Add the Reset button with a callback to reset the session state and delete the image
st.button('Reset', on_click=reset_session)

# Handle session initialization and reset logic
if st.session_state.reset:
    # Display a reset message and allow a new upload
    st.info("The session has been reset. Upload a new image.")
    # Reset session flag, ready for new upload
    st.session_state.reset = False
    initiate_session()
else:
    initiate_session()
