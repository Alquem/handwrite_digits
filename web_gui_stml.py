import tensorflow as tf
from tensorflow import keras
import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

def classify(image, model):
    # convert image to (224, 224)
    image = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # set model input
    data = np.ndarray(shape=(1, 28, 28, 1), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    res = model.predict(data)[0]

    #res = model.predict([img])[0]
    
    return np.argmax(res), max(res)

def main():
    # set title
    st.title('Digit classification')
    
    # set header
    st.header('Please draw a single digit number')
    
    # Specify canvas parameters in application
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
    )
    
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == 'point':
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    
        
    
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=150,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )
    
    
    
    
    # load classifier
    model = load_model('model_mnist.h5')
    
    if canvas_result.image_data is not None:
    
        st.image(canvas_result.image_data, use_column_width=True)
    
        # classify image
        class_name, conf_score = classify(image, model)
    
        # write classification
        st.write("## {}".format(class_name))
        st.write("### score: {}%".format(int(conf_score * 100)))



    
if __name__ == '__main__':
    main()
