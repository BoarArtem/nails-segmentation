import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np

st.title("Nails segmentation")

def load_image(img_path: str = "runs/segment/train-4/weights/best.pt"):
    return YOLO(img_path)

model = load_image()

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.success("Image successfully uploaded")

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    result = model(image_np)

    annotated_frame = result[0].plot(boxes=False, labels=False)

    st.image(
        annotated_frame,
        caption="Segmentation Result",
        use_container_width=True,
    )


