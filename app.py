import gradio as gr
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# --- 1. Define Column Names ---
# These are the 30 feature columns your model expects, in order.
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# --- 2. Load Saved Objects ---
# We load these once when the app starts.
try:
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    with open('power_transformer.pkl', 'rb') as f:
        pt = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    model = keras.models.load_model('my_model.keras')
    
    print("All models and pre-processors loaded successfully.")

except FileNotFoundError as e:
    print(f"Error: Could not find {e.filename}.")
    print("Please make sure 'label_encoder.pkl', 'power_transformer.pkl', 'scaler.pkl', and 'my_model.keras' are in the same directory.")
    # We can't run the app if files are missing.
    le = None
    pt = None
    scaler = None
    model = None

# --- 3. Define Prediction Function ---
def predict(input_df):
    """
    Takes a 1-row Pandas DataFrame of 30 features, processes it,
    and returns a prediction for HighlightedText.
    """
    if not all([le, pt, scaler, model]):
        # Return format for HighlightedText
        return [("Error: Models not loaded. Check console.", "ERROR")]
        
    try:
        # Convert DataFrame to numpy array
        input_data = input_df.values
        
        # --- FIX: Check for and correct transposed input ---
        # The Gradio component sometimes passes (30, 1) instead of (1, 30).
        # We detect this and transpose it back.
        if input_data.shape[0] == 30 and input_data.shape[1] == 1:
            print("DEBUG: Transposing (30, 1) input to (1, 30)")
            input_data = input_data.T  # Transpose from (30, 1) to (1, 30)
        
        # Add a final check for the correct shape
        if input_data.shape[0] != 1 or input_data.shape[1] != 30:
            msg = f"Error: Input data has wrong shape. Expected (1, 30), got {input_data.shape}"
            print(msg)
            return [(msg, "ERROR")]
        # --- End of Fix ---

        # --- Handle PowerTransformer ---
        # Your 'pt' was trained on 31 columns ('diagnosis' + 30 features).
        # We must re-create this shape for pt.transform() to work.
        
        # Create a dummy 'diagnosis' column (shape (1, 1))
        dummy_diag = np.zeros((input_data.shape[0], 1))
        
        # Prepend the dummy column to our 30 features -> shape (1, 31)
        data_for_pt = np.hstack([dummy_diag, input_data])
        
        # Apply PowerTransformer
        transformed_data = pt.transform(data_for_pt)
        
        # Remove the transformed dummy 'diagnosis' column (first column)
        features_for_scaler = transformed_data[:, 1:] # shape (1, 30)
        
        # --- Handle StandardScaler ---
        # Apply the scaler (which was fit on pt-transformed data)
        scaled_features = scaler.transform(features_for_scaler)
        
        # --- Make Prediction ---
        prediction = model.predict(scaled_features)
        
        # Process the model's output (sigmoid)
        pred_int = (prediction > 0.5).astype(int).reshape(-1)
        
        # --- Decode Label ---
        final_label = le.inverse_transform(pred_int)[0]
        
        # Return a user-friendly, color-coded string for HighlightedText
        if final_label == 'M':
            return [("Malignant", "NEGATIVE")]
        else:
            return [("Benign", "POSITIVE")]
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        # Return format for HighlightedText
        return [(f"Error: {e}", "ERROR")]

# --- 4. Create an Example for the Interface ---
# This is the "Malignant" example, used as the default for the input grid.
example_data_malignant = [[
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]]
# Create the DataFrame for the default grid content (Example 1)
example_df_malignant = pd.DataFrame(data=example_data_malignant, columns=FEATURE_NAMES)

# This is the new "Benign" example row you provided (just the 30 features)
example_data_benign_row = [
    13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
    0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
    15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
]
# Create the DataFrame for the second example
example_df_benign = pd.DataFrame(data=[example_data_benign_row], columns=FEATURE_NAMES)


# --- 5. Launch Gradio Interface ---
# Added theme=gr.themes.Glass() for a modern UI
with gr.Blocks(theme=gr.themes.Glass()) as iface:
    gr.Markdown(
        """
        # Breast Cancer Diagnosis Predictor
        Enter the 30 cell nucleus features below to predict whether the diagnosis
        is Malignant (M) or Benign (B).
        This app uses a Keras neural network model with data pre-processing
        (PowerTransform and StandardScaler).
        """
    )
    
    with gr.Row():
        input_grid = gr.Dataframe(
            headers=FEATURE_NAMES,
            label="Input Features",
            value=example_df_malignant  # This correctly sets the default value
        )

    with gr.Row():
        predict_btn = gr.Button("Predict", variant="primary")
    
    with gr.Row():
        # Changed from gr.Label to gr.HighlightedText for colored output
        output_result = gr.HighlightedText(
            label="Prediction",
            color_map={"POSITIVE": "green", "NEGATIVE": "red", "ERROR": "grey"},
            show_label=True
        )

    predict_btn.click(
        fn=predict,
        inputs=input_grid,
        outputs=output_result # Changed output to the new component
    )
    
    # Updated to pass a list of lists, where each inner list contains
    # one DataFrame object for the input_grid component.
    gr.Examples(
        examples=[[example_df_malignant], [example_df_benign]],
        inputs=input_grid,
        examples_per_page=2
    )

if __name__ == "__main__":
    iface.launch()

