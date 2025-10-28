# 🧠 Breast Cancer Diagnosis Predictor (ANN)

This project uses an **Artificial Neural Network (ANN)** built with **Keras/TensorFlow** to predict whether a breast cancer diagnosis is **Malignant (M)** or **Benign (B)** based on 30 cell nucleus features.

The trained model and pre-processing pipeline (`PowerTransformer`, `StandardScaler`, `LabelEncoder`) are saved and served through an interactive web application built with **Gradio**.

---

## 🚀 Technology Stack

- **Model Training:** TensorFlow (Keras), Scikit-learn, Pandas, NumPy, Jupyter  
- **Web Application:** Gradio  
- **Data Pre-processing:** `LabelEncoder`, `PowerTransformer`, `StandardScaler`  
- **Model Saving:** `pickle` (for pre-processors), `model.save()` (for Keras model)

---

## 📂 Project Structure

```

.
├── app.py                  # The Gradio web application script
├── code.ipynb              # Jupyter Notebook for data processing and model training
├── requirements.txt        # Python dependencies
│
├── my_model.keras          # Saved trained Keras model
├── label_encoder.pkl       # Saved LabelEncoder (for diagnosis 'M'/'B')
├── power_transformer.pkl   # Saved PowerTransformer (for fixing data skew)
└── scaler.pkl              # Saved StandardScaler (for feature scaling)

````

---

## 🧩 How to Run the Web App

Follow these steps to set up and run the Gradio prediction interface on your local machine.

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
````

### Step 2: Set Up a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv myenv

# Activate the environment
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate
```

### Step 3: Install Dependencies

Create a file named `requirements.txt` (if not already present) with the following content:

```text
gradio==4.44.0
tensorflow==2.16.1
scikit-learn==1.5.1
pandas==2.2.2
numpy==1.26.4
```

Then install all dependencies:

```bash
pip install -r requirements.txt
```

### Step 4: Run the Gradio App

Make sure all `.pkl` files and `my_model.keras` are in the same directory as `app.py`.

```bash
python app.py
```

### Step 5: Open in Browser

After running, you’ll see a local URL in your terminal such as:

```
http://127.0.0.1:7860
```

Open it in your browser to use the application.

---

## 🔁 (Optional) Re-training the Model

If you want to re-train or modify the ANN model:

1. Ensure all dependencies from `requirements.txt` are installed.
2. Install Jupyter (if not installed):

   ```bash
   pip install notebook
   ```
3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```
4. Open `code.ipynb` and run all cells.
   This will retrain the model and overwrite:

   ```
   label_encoder.pkl
   power_transformer.pkl
   scaler.pkl
   my_model.keras
   ```

---

## 💡 Credits

Developed using:

* **Python 3.10+**
* **TensorFlow / Keras**
* **Scikit-learn**
* **Gradio**

---

### 🧾 License

This project is released under the **MIT License**.
Feel free to modify and use it for learning or research purposes.

```

---

Would you like me to make it **GitHub-badge ready** (with TensorFlow/Gradio badges and deployment instructions for Hugging Face or Render)? It’ll make your README look professional for portfolio/demo purposes.
```
