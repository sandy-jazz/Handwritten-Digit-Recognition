Handwritten Digit Recognition
A small web app that recognizes a handwritten digit (0-9) from an image. You can upload a photo or take one with your camera, and a CNN model trained on the MNIST dataset predicts which digit it is. The app is built with Python, TensorFlow/Keras and Streamlit.

What it does
Upload a PNG/JPG/JPEG image or take a photo with the camera
Cleans the image so it looks like an MNIST sample (grayscale, inverted, cropped, centred, resized to 28 x 28)
Predicts the digit with the trained CNN model
Shows the predicted digit, the confidence and the second best guess
Shows the probability of all 10 digits as bars
Shows the original image next to what the model actually sees
How it works
The image is opened and transparent backgrounds are replaced with white.
It is converted to grayscale.
MNIST digits are white on a black background, so if the image has dark ink on white paper it is inverted.
The empty space around the digit is cropped.
The digit is resized to fit inside 20 x 20 and pasted in the centre of a 28 x 28 black image (same as MNIST).
Pixel values are scaled to 0-1 and passed to the CNN.
The model returns 10 probabilities and the highest one is the predicted digit.
Project structure
project-folder/
├── app.py                     # the Streamlit app (use your own file name)
├── backgroundimage.jpg        # optional, background of the page
├── requirements.txt
├── README.md
└── models/
    └── digit_cnn_model.keras  # trained CNN model
If backgroundimage.jpg is missing, the app uses a simple gradient background instead. The model file is required.

Requirements
Python 3.9 to 3.12 (TensorFlow does not support the newest Python versions right away)
Streamlit 1.40 or newer (the app uses st.container(key=...) and use_container_width)
TensorFlow, NumPy and Pillow
requirements.txt:

streamlit>=1.40
tensorflow
numpy
pillow
How to run
Open a terminal in the project folder.

(Optional) Create a virtual environment:

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
Install the packages:

pip install -r requirements.txt
Make sure the trained model is at models/digit_cnn_model.keras.

Start the app:

streamlit run app.py
Open the link shown in the terminal (usually http://localhost:8501).

Tips for good predictions
Write only one digit in the image
Use dark ink on plain white paper
Use a thick pen or marker, thin lines are often predicted wrongly
Keep the digit in the middle with good lighting and no shadows
If the confidence is low, try again with a clearer photo
Troubleshooting
Problem	Fix
"Could not load the model"	Check that models/digit_cnn_model.keras exists and the path is correct
Text is hard to see (white on white)	Do a hard refresh in the browser (Ctrl + Shift + R). The theme colours are set in the CSS inside the app
Wrong prediction	Retake the photo with a thicker stroke and a plain background
Camera not working	Allow camera permission in the browser
TensorFlow install error	Use a supported Python version (3.9 to 3.12)
Tech used
Python, TensorFlow, Keras, CNN, Streamlit, NumPy, Pillow, MNIST dataset

Limitations
Works for single digits only, not full numbers or words
Trained on MNIST, so very unusual handwriting styles or messy backgrounds can reduce accuracy
Author
Your name here
