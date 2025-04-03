import streamlit as st
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

UPLOAD_FOLDER = "uploads"
OUTPUT_IMAGE = "output.png"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

st.title("CUDA Image Filtering")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    input_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    filter_options = {
        "Blur": 1,
        "Sharpen": 2,
        "Outline": 3,
        "Color Inversion": 4,
        "Black & White": 5,
    }

    selected_filter = st.selectbox("Select Filter", list(filter_options.keys()))
    intensity = st.slider("Select Intensity (Only for Blur & Sharpen)", 1, 10, 5) if selected_filter in ["Blur", "Sharpen"] else 1

    st.image(input_path, caption="Original Image", use_column_width=False, width=300)

    if st.button("Apply Filter"):
        command = f"image_filter.exe {input_path} {filter_options[selected_filter]} {intensity}"
        subprocess.run(command, shell=True, capture_output=True, text=True)

        if os.path.exists(OUTPUT_IMAGE):
            col1, col2 = st.columns(2)
            with col1:
                st.image(input_path, caption="Original", use_column_width=True)
            with col2:
                st.image(OUTPUT_IMAGE, caption="Processed", use_column_width=True)

        st.success("Processing complete!")
