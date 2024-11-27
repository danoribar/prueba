import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("/mount/src/prueba/cnn_model.h5")
label_map = {0:"cat",1:"Dog"}

def preprocess_iamge(image):
    image = image.resize((128,128))
    image = np.array(image) / 255
    image = np.expand_dims(image,axis=0)
    return image

st.title("Clasificación CNN")
st.write("Inserte imagen")

upload_image = st.file_uploader("Elija imagen..",type=["jpg","png"])

if upload_image is not None:
    image = Image.open(upload_image)
    st.image(image,caption="Imagen cargada",use_column_width=True)

    image_preprocessed = preprocess_iamge(image=image)


    prediction = model.predict(image_preprocessed)
    predicted_class = np.argmax(prediction)
    predicted_label = label_map[predicted_class]

    st.write(f"Predicción: {predicted_label}")
