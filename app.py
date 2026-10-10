import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="AI Pet Breed Classifier", layout="centered")
st.title("🐾 AI Pet Breed Classifier")
st.write("Upload an image of a cat or dog, and our deep learning model (ResNet50) will classify its breed!")

# 2. Cache the model load so it runs ultra-fast
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("pet_classifier_model.keras")

try:
    model = load_my_model()
except Exception as e:
    st.error("Could not load the model. Ensure pet_classifier_model.keras is in the same folder.")

# 3. Label Map (Matches cell [8] of your notebook)
idx_to_label = {
    0: 'Abyssinian', 1: 'Bengal', 2: 'Birman', 3: 'Bombay', 
    4: 'British Shorthair', 5: 'Egyptian Mau', 6: 'American Bulldog', 
    7: 'American Pit Bull Terrier', 8: 'Basset Hound', 9: 'Beagle', 
    10: 'Boxer', 11: 'Chihuahua', 12: 'English Cocker Spaniel', 
    13: 'English Setter', 14: 'German Shorthaired', 15: 'Great Pyrenees'
}

# 4. File Uploader UI Widget
uploaded_file = st.file_uploader("Choose a JPEG or PNG image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing image features..."):
        # 5. Image Preprocessing (Matches cell [10] of your notebook)
        IMAGE_SIZE = (224, 224)
        
        img_array = img_to_array(image, dtype='uint8')
        resized_img = tf.image.resize_with_pad(img_array, *IMAGE_SIZE)
        final_image = np.array(resized_img).astype('uint8')
        
        # Add batch dimension: shape becomes (1, 224, 224, 3)
        input_batch = np.expand_dims(final_image, axis=0)
        
        # 6. Model Prediction
        predictions = model.predict(input_batch)
        predicted_class_idx = np.argmax(predictions)
        confidence_score = float(np.max(predictions)) * 100
        
        # 7. Output Results Dynamically
        predicted_breed = idx_to_label.get(predicted_class_idx, "Unknown Breed")
        
        st.success(f"### Result: **{predicted_breed}**")
        st.info(f"💡 **Model Confidence:** {confidence_score:.2f}%")
