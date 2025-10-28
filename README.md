# Breast Cancer Diagnosis Predictor (ANN)

This project uses an Artificial Neural Network (ANN) built with Keras/TensorFlow to predict whether a breast cancer diagnosis is Malignant (M) or Benign (B) based on 30 cell nucleus features.

The trained model is served through a user-friendly web interface created with Gradio.

# Application Demo

The Gradio app provides a simple interface to enter the 30 required features for prediction.
APP:
After clicking 'Predict', the model will return the diagnosis, color-coded for clarity (Green for Benign, Red for Malignant).

Project Structure

This repository contains the following key files:

code.ipynb: The Jupyter Notebook containing all steps for data loading, cleaning, pre-processing (LabelEncoding, PowerTransform, Scaling), model training, and evaluation.

app.py: The Python script that loads the pre-trained model and pre-processors to launch a live Gradio web application.

requirements.txt: A list of all Python libraries required to run the application.

label_encoder.pkl: The saved LabelEncoder object (for diagnosis).

power_transformer.pkl: The saved PowerTransformer object (for fixing data skew).

scaler.pkl: The saved StandardScaler object (for feature scaling).

my_model.keras: The saved, trained Keras neural network model.

image_8c335e.png, image_8c337d.png, image_8c33a0.png: The screenshot images used in this README.

# How to Run the Web App

Follow these steps to set up and run the Gradio prediction interface on your local machine.

Step 1: Set Up a Virtual Environment (Recommended)

It's best practice to create a virtual environment to manage project dependencies.

# Create a virtual environment
python -m venv myenv

# Activate the environment
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate


Step 2: Install Dependencies

With your virtual environment active, install all the required libraries from requirements.txt.

pip install -r requirements.txt


Step 3: Run the Gradio App

Make sure all the .pkl files, the my_model.keras file, and the image files are in the same directory as app.py.

python app.py


Step 4: Open in Browser

The terminal will display a local URL, typically http://122.0.0.1:7860. Open this link in your web browser to use the application.

(Optional) How to Re-train the Model

If you want to re-train the model from scratch:

Ensure you have all the libraries from requirements.txt installed.

You will also need jupyter to run the notebook: pip install notebook.

Launch Jupyter: jupyter notebook.

Open code.ipynb and run all the cells. This will re-generate the label_encoder.pkl, power_transformer.pkl, scaler.pkl, and my_model.keras files with the newly trained data.