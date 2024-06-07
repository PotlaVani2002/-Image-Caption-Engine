import streamlit as st
import numpy as np
import pickle
import base64
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from PIL import Image
import os
from io import BytesIO

# Paths
BASE_DIR = 'E:/Projects/-Image-Caption-Engine/'
WORKING_DIR = 'E:/Projects/-Image-Caption-Engine/Saved_models/'

# Load the features and captions
with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()

# Create mapping of image to captions
mapping = {}
for line in captions_doc.split('\n'):
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

# Clean the captions
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i].lower()
            caption = caption.replace('[^A-Za-z]', '')
            caption = caption.replace('\s+', ' ')
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption

clean(mapping)

# Prepare tokenizer
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in all_captions)

# Load the features
with open(os.path.join(BASE_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)

# Load the model
model = load_model(os.path.join(WORKING_DIR, 'model.h5'))
vgg_model = load_model(os.path.join(WORKING_DIR, 'vgg_model.h5'))

# Function to convert integer to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to generate caption
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    in_text = in_text.replace('startseq', '').replace('endseq', '').strip()
    return in_text

# Add CSS styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background: #4CAF50;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background: #45a049;
    }
    .stFileUploader>div>div>button {
        background: #008CBA;
        color: white;
    }
    .stFileUploader>div>div>button:hover {
        background: #007bb5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title("Image Caption Engine")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
image_size = st.slider("Select image size", 100, 300, 150)

if uploaded_files:
    st.write("Generating captions for uploaded images...")
    image_elements = []

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image_resized = image.resize((224, 224))
        image_array = img_to_array(image_resized)
        image_array = image_array.reshape((1, 224, 224, 3))
        image_array = preprocess_input(image_array)
        feature = vgg_model.predict(image_array, verbose=0)
        caption = predict_caption(model, feature, tokenizer, max_length)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        image_elements.append((caption, img_base64))
    cols_per_row = 3
    rows = len(image_elements) // cols_per_row + (1 if len(image_elements) % cols_per_row > 0 else 0)
    
    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col, (caption, img_base64) in zip(cols, image_elements[row * cols_per_row: (row + 1) * cols_per_row]):
            with col:
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <img src="data:image/jpeg;base64,{img_base64}" style="width:{image_size}px;height:auto;"/>
                        <p>{caption}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
