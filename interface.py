import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image

st.title("Cell Nuclei Segmentation")
st.write("This application segments cell nuclei in microscopy images using a pre-trained U-Net model.")
st.write("\n\n")
st.write("Upload a microscopy image (JPEG or PNG) to see the segmentation results.")
img=st.file_uploader("Upload Image", type =["jpg", "jpeg", "png"])


if img is not None:
    st.write("Image uploaded successfully!")
    st.image(img)


st.write("\n\n")
pred_init = st.button("Segment Nuclei")

if pred_init and img is not None:
    def run_segmentation():
        IMG_WIDTH = 128
        IMG_HEIGHT = 128
        IMG_CHANNELS = 3
        model = keras.models.load_model("model-for-unet.h5", compile=False)

        # Preprocess the image: resize and normalize
        processed_img = Image.open(img).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
        img_array = np.array(processed_img, dtype=np.float32)
        # Normalize to [0,1] range as expected by the model
        img_array = img_array.reshape((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        
        # Make prediction
        pred = model.predict(img_array, verbose=0)
        
        # Process prediction: squeeze to remove batch dimension and convert to proper display format
        segmented_image = pred[0].squeeze()  # Shape: (128, 128)
        
        # Apply threshold and convert to uint8 for display (0-255 range)
       
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.image(processed_img, caption="Input Image", use_container_width=True)
        with col2:
            st.image(segmented_image, caption="Segmented Nuclei", use_container_width=True)
            
        st.write("Segmentation completed!")
        
    run_segmentation()

