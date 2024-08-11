import streamlit as st
from PIL import Image
from fastai.vision.all import *
import io

# Load the Fastai learner model
learner = load_learner('models/resnetmodel.pkl')

# Dinosaur descriptions
dino_descriptions = {
    "parasaurolophus": "Parasaurolophus was a herbivorous dinosaur with a distinctive long, curved crest on its head, used for display and communication.",
    "spinosaurus": "Spinosaurus was a large, semi-aquatic dinosaur known for its elongated, sail-like structure on its back and a crocodile-like snout.",
    "stegosaurus": "Stegosaurus was a herbivorous dinosaur characterized by its large bony plates along its back and the spikes on its tail, known as the thagomizer.",
    "triceratops": "Triceratops was a herbivorous dinosaur with three distinctive facial horns and a large bony frill protecting its neck.",
    "tyrannosaurus-rex": "Tyrannosaurus Rex, often referred to as T. rex, was one of the largest and most fearsome carnivorous dinosaurs, known for its powerful jaws and tiny arms."
}

# Set the title and description with dark theme styling
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #e0e0e0;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #bb86fc;
    }
    .description {
        font-size: 20px;
        font-weight: 300;
        color: #b0b0b0;
    }
    .button {
        font-size: 18px;
        font-weight: bold;
        background-color: #bb86fc;
        color: #121212;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .button:hover {
        background-color: #9a67ea;
    }
    .prediction {
        font-size: 22px;
        font-weight: bold;
        color: #03dac6;
    }
    .confidence {
        font-size: 20px;
        color: #ffeb3b;
    }
    .not-confident {
        font-size: 20px;
        color: #cf6679;
    }
    .description-text {
        font-size: 18px;
        color: #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🦖 AI-Powered Dinosaur Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Explore the power of AI as it classifies dinosaurs like Parasaurolophus, Spinosaurus, Stegosaurus, Triceratops, and T. rex. Our tool leverages advanced deep learning to identify these prehistoric giants with impressive accuracy.</div>', unsafe_allow_html=True)

# File uploader for user to upload an image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load and process the image
    image = Image.open(io.BytesIO(uploaded_image.read()))

    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Resize the image to 224x224 pixels
    resized_image = image.resize((224, 224))

    # Display the uploaded and resized image
    st.image(resized_image, caption="Uploaded Image (224x224)", use_column_width=True)

    # Convert the resized image to a format suitable for the model
    img_fastai = PILImage.create(resized_image)

    if st.button('Predict', key='predict_button', help='Click to make a prediction', use_container_width=True):
        # Make a prediction using the loaded learner model
        pred, pred_idx, probs = learner.predict(img_fastai)

        # Display the prediction if the probability is above 70%
        if probs[pred_idx].item() * 100 > 70:
            st.markdown('<div class="prediction">Prediction:</div>', unsafe_allow_html=True)
            st.write(f"**Predicted Dinosaur**: {pred}")
            st.markdown(f'<div class="confidence">Confidence: {probs[pred_idx].item() * 100:.2f}%</div>', unsafe_allow_html=True)
            
            # Display the description for the predicted dinosaur
            description = dino_descriptions.get(pred, "Description not available.")
            st.markdown(f'<div class="description-text">{description}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction">Prediction:</div>', unsafe_allow_html=True)
            st.markdown('<div class="not-confident">The model is not detecting a dinosaur with sufficient confidence.</div>', unsafe_allow_html=True)
else:
    st.text("Please upload an image to start the classification.")
