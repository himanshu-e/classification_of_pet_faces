# 🐾 Deep Learning Pet Breed Classifier (ResNet50)

## 📌 Business Overview & Objective
Manual classification of animal breeds in veterinary clinics, pet shelter onboarding, and animal insurance claims handling is slow and error-prone. This end-to-end computer vision project builds an automated classification pipeline using Deep Learning, cutting down check-in evaluation times from minutes to milliseconds.

## 🛠️ Tech Stack & Skills Demonstrated
* **Deep Learning Framework:** TensorFlow / Keras (Functional API, Sequential Model API).
* **Transfer Learning Architecture:** ResNet50 (pre-trained on ImageNet) to extract high-level feature extractions without computational bottlenecks.
* **Data Pipelines:** Integrated real-time spatial transformations including `RandomFlip` and `RandomRotation` for robust regularization.
* **Web UI Engineering:** Cloud deployment via Streamlit.

## 📊 Core Performance Metrics
* Implements **padding-preserved aspect ratio scaling** to prevent geometric distortion during image input.
* Final evaluations metrics scored across 16 highly challenging target classes containing complex geometric inter-class variations (e.g., distinguishing American Bulldog vs. Pit Bull Terrier).
