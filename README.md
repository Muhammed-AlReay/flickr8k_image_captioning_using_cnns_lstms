# 🖼 Image-Caption-Generator-Using-Deep-Learning-Image-Captioning-Using-CNN-LSTM
A Deep Learning Project using the Flickr8k dataset

## 📌 Overview
This project generates natural language captions for images using a hybrid deep learning architecture that combines Convolutional Neural Networks (CNNs) and Long Short-Term Memory networks (LSTMs).
It’s trained on the Flickr8k dataset, which contains 8,000 images, each annotated with five human-generated captions.

## ✅ Features
Extracts image features using pre-trained CNN (e.g., InceptionV3 or VGG16).

Generates captions using LSTM-based sequence models.

Supports training from scratch or using extracted features.

Evaluation using the BLEU Score for caption quality.

Includes a working notebook and Python script for easy experimentation.

Sample results and screenshots provided.

## 🧠 Technologies Used
Python

TensorFlow / Keras

NumPy, Pandas

Matplotlib, Seaborn

NLTK (for text preprocessing)

Jupyter Notebook

Pre-trained CNN models (e.g., InceptionV3)

## 📁 File Structure
csharp
Copy
Edit
flickr8k_image_captioning_using_cnns_lstms/
│
├── main.py                              # Main script
├── flickr8k-image-captioning-using-cnns-lstms.ipynb  # Jupyter notebook
├── img.png, img_1.png, ...              # Sample test images
├── Screenshot *.png                     # Output screenshots
├── README.md                            # Project documentation
🧪 How It Works
CNN (e.g., InceptionV3) encodes the image into a fixed-length feature vector.

The LSTM model takes this feature vector and a sequence of words to predict the next word.

During inference, the model predicts the caption one word at a time using the previously generated words.

## ▶ Usage
Option 1: Run the Notebook
Open the Jupyter notebook:

bash
Copy
Edit
jupyter notebook flickr8k-image-captioning-using-cnns-lstms.ipynb
Option 2: Run Python Script
bash
Copy
Edit
python main.py
Make sure your main.py is configured with paths to your model weights and test images.

## 🖼 Sample Output
"A man riding a bike on the street"

"Two children playing with a ball in a park"

(Check the screenshots in the repo for more examples)

## 📦 Dataset
The model is trained using the Flickr8k dataset:

8,000 images

5 captions per image


## 🔧 Future Improvements
Add an attention mechanism for better caption quality.

Switch to Transformer-based architectures (e.g., ViT + GPT).

Deploy via Streamlit or Flask web app.

Add beam search decoding.

## 👤 Author
Muhammed-AlReay
