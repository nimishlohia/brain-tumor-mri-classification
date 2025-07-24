import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")
MODEL_PATH = "resnet50_brain_tumor_model.h5"  # Place model in same folder
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
LAST_CONV_LAYER_NAME = "conv5_block3_out"  # ResNet50 last conv layer

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_brain_model():
    model = load_model(MODEL_PATH, compile=False)
    return model

model = load_brain_model()

# -------------------- HELPERS --------------------
def preprocess(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # img_array must be float32 tensor
    if isinstance(img_array, np.ndarray):
        img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array, training=False)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]                              # (H, W, C)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # ReLU + normalize
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), int(class_idx.numpy())

def overlay_heatmap(original_img: Image.Image, heatmap, alpha=0.4):
    img = np.array(original_img.convert("RGB"))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlayed = heatmap_colored * alpha + img
    overlayed = np.uint8(overlayed)
    return overlayed

def plot_probs(probs):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.barh(CLASS_NAMES, probs)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    st.pyplot(fig)

# -------------------- UI --------------------
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload a brain MRI image to detect tumor type with **ResNet50 + Grad-CAM**.")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])
alpha = st.sidebar.slider("Grad-CAM Overlay Alpha", 0.0, 1.0, 0.4, 0.05)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            arr = preprocess(img)
            preds = model.predict(arr)
            probs = preds[0]
            pred_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]

            heatmap, _ = make_gradcam_heatmap(arr, model, LAST_CONV_LAYER_NAME)
            overlayed = overlay_heatmap(img, heatmap, alpha=alpha)

        st.success(f"Prediction: **{pred_class}** ({probs[pred_idx]*100:.2f}%)")
        st.write("### Class Probabilities")
        plot_probs(probs)

        st.write("### Grad-CAM Visualization")
        st.image(overlayed, caption="Model Attention", use_column_width=True)
