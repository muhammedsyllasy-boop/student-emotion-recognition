import os
import json
import streamlit as st
import pandas as pd
from PIL import Image
from predict import predict_expression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics.json")

st.set_page_config(page_title="Student Expression Recognition", page_icon="🙂", layout="wide")

st.title("🙂 Student Facial-Expression Recognition")
st.write(
    "A deep-learning demonstration that classifies visible facial-expression patterns "
    "using a CNN trained on FER-2013."
)
st.warning(
    "Research prototype only: facial expressions do not prove a person's internal emotion. "
    "Do not use predictions to grade, diagnose, discipline, or make decisions about students."
)

@st.cache_resource
def load_model():
    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("The trained model was not found.")
    st.markdown(
        "### First-time setup\n"
        "1. Make sure a FER-2013 dataset is available in the `data` folder.\n"
        "2. In your terminal run `python train.py`.\n"
        "3. When training finishes, restart this app."
    )
    st.stop()

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("Upload an image")
    uploaded = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"]
    )
    st.caption("For best results, upload a clear, front-facing, single-face image. "
               "The prototype resizes the whole image; it does not automatically detect/crop faces.")

with right:
    st.subheader("Model performance")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        st.metric("Test accuracy", f"{metrics.get('test_accuracy', 0):.1%}")
        st.metric("Test samples", f"{metrics.get('test_samples', 0):,}")
        st.caption("These values are shown only after you run train.py on your dataset.")
    else:
        st.info("Train the model to generate measured test metrics.")

image = None
if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", width=300)
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
        st.info("Try a valid JPG or PNG image.")
        image = None
else:
    st.info("Upload an image to enable facial-expression prediction.")

predict_clicked = st.button(
    "Predict facial expression",
    type="primary",
    use_container_width=True,
    disabled=image is None,
)

if predict_clicked and image is not None:
    try:
        with st.spinner("Loading the model and analysing the image. The first prediction may take a little longer..."):
            model = load_model()
            result = predict_expression(model, image)
        st.success(f"Predicted expression: {result['label']}")
        st.metric("Model confidence", f"{result['confidence']:.1%}")
        st.caption("Confidence is the model's score for this label, not the probability that the person truly feels it.")
        scores = pd.DataFrame({
            "Expression": list(result["probabilities"].keys()),
            "Model score": list(result["probabilities"].values())
        }).sort_values("Model score", ascending=True)
        st.subheader("Scores for all expression categories")
        st.bar_chart(scores.set_index("Expression"))
    except Exception as exc:
        st.error(f"Could not generate a prediction: {exc}")
        st.info("Check that training completed successfully and that the saved model is valid.")

st.divider()
st.subheader("About the project")
st.write(
    "Architecture: convolutional layers, batch normalisation, max pooling, dropout, "
    "global average pooling, and a seven-class Softmax output. Dataset: FER-2013 "
    "(48 × 48 grayscale images)."
)
st.write("Use only images you have permission to process. Avoid uploading sensitive images or images of people who have not consented.")
