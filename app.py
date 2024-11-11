from flask import Flask, request, render_template, jsonify, send_from_directory
import joblib
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import logging
import tensorflow as tf
from sklearn.preprocessing import PowerTransformer
classes = ['Apple_Apple Black rot', 'Apple_Apple Healthy', 'Apple_Apple Scab',
       'Apple_Cedar apple rust', 'Bell pepper_Bell pepper Bacterial spot',
       'Bell pepper_Bell pepper Healthy', 'Cherry_Cherry Healthy',
       'Cherry_Cherry Powdery mildew', 'Citrus_Citrus Black spot',
       'Citrus_Citrus Healthy', 'Citrus_Citrus canker',
       'Citrus_Citrus greening', 'Corn_Corn Common rust',
       'Corn_Corn Gray leaf spot', 'Corn_Corn Healthy',
       'Corn_Corn Northern Leaf Blight', 'Grape_Grape Black Measles',
       'Grape_Grape Black rot', 'Grape_Grape Healthy',
       'Grape_Grape Isariopsis Leaf Spot', 'Peach_Peach Bacterial spot',
       'Peach_Peach Healthy', 'Potato_Potato Early blight',
       'Potato_Potato Healthy', 'Potato_Potato Late blight',
       'Strawberry_Strawberry Healthy',
       'Strawberry_Strawberry Leaf scorch',
       'Tomato_Tomato Bacterial spot', 'Tomato_Tomato Early blight',
       'Tomato_Tomato Healthy', 'Tomato_Tomato Late blight',
       'Tomato_Tomato Leaf Mold', 'Tomato_Tomato Mosaic virus',
       'Tomato_Tomato Septoria leaf spot', 'Tomato_Tomato Spider mites',
       'Tomato_Tomato Target Spot',
       'Tomato_Tomato Yellow Leaf Curl Virus']
# Initialize the Flask app
app = Flask(__name__, template_folder=r'C:\Users\Target\OneDrive\Desktop\AgricultureProjectWebsite\templates')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load models and preprocessors
try:
    crop_model = joblib.load('models/Extra_Tree_model (1).pkl')
    encoder_1 = joblib.load("models/Ordinal_Encoder.pkl")
    encoder_2 = joblib.load("models/label_Encoder (1).pkl")
    scaler = joblib.load("models/FeatureScaler.pkl")
    disease_model_path = 'models/Effcient_Net_Model.pth'
    disease_model = models.efficientnet_b0(weights=None, num_classes=37)
    disease_model.classifier[1] = torch.nn.Linear(disease_model.classifier[1].in_features, 37)
    disease_model.load_state_dict(torch.load(disease_model_path, map_location='cpu'))
    disease_model.eval()
    weather_model_path = r"C:\Users\Target\OneDrive\Desktop\AgricultureProjectWebsite\models\weather_prediction_model.h5"
    weather_model = tf.keras.models.load_model(weather_model_path)
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    exit(1)


# Transformation functions
def feature_engineering(data):
    data['NP_Ratio'] = data['Nitrogen'] / data['Phosphorus']
    data['NK_Ratio'] = data['Nitrogen'] / data['Potassium']
    data['PK_Ratio'] = data['Phosphorus'] / data['Potassium']
    data["NPK_Average"] = (data['Nitrogen'] + data["Phosphorus"] + data["Potassium"]) / 3
    data["Temp_Humididty_Index"] = data["Temperature"] * data['Humidity']
    data["Rainfall_Humidity_Index"] = data['Rainfall'] * data['Humidity']
    return data


def log_transform(data, log_columns):
    for col in log_columns:
        data[f'Log_{col}'] = np.log1p(data[col])
        data.drop(col, axis=1, inplace=True)
    return data


def sqrt_transform(data, sq_columns):
    for col in sq_columns:
        data[f'SQ_{col}'] = np.sqrt(data[col].clip(lower=0))
        data.drop(col, axis=1, inplace=True)
    return data


def power_transform(data, pt_columns):
    transformers = {}
    for col in pt_columns:
        if data[col].nunique() > 1:
            pt = PowerTransformer(method='yeo-johnson')
            transformers[col] = pt.fit(data[[col]])
            data[f'PT_{col}'] = transformers[col].transform(data[[col]])
        else:
            data[f'PT_{col}'] = data[[col]]
        data.drop(col, axis=1, inplace=True)
    return data


# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Routes for templates
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/crop-recommendation.html')
def crop_recommendation():
    return render_template('crop-recommendation.html')


@app.route('/web.html')
def plant_diseases():
    return render_template('web.html')


@app.route('/xx.html')
def weather_forecasting():
    return render_template('xx.html')


# Crop recommendation prediction route
@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        # Get form data and process
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph_value = float(request.form['ph_value'])
        rainfall = float(request.form['rainfall'])

        user_data = pd.DataFrame({
            'Nitrogen': [nitrogen],
            'Phosphorus': [phosphorus],
            'Potassium': [potassium],
            'Temperature': [temperature],
            'Humidity': [humidity],
            'pH_Value': [ph_value],
            'Rainfall': [rainfall]
        })

        user_data = feature_engineering(user_data)
        user_data['PH_Categories'] = user_data['pH_Value'].apply(
            lambda x: "Acidic" if x < 5.5 else "Neutral" if x <= 7.5 else "Alkaline")
        user_data['PH_Cat'] = encoder_1.transform(user_data[['PH_Categories']])
        user_data.drop(['PH_Categories'], axis=1, inplace=True)

        log_columns = ['Phosphorus', 'Humidity', 'Rainfall', 'NK_Ratio', 'PK_Ratio', 'NPK_Average',
                       'Rainfall_Humidity_Index']
        pt_columns = ['Potassium', 'NP_Ratio']
        sq_columns = ['Nitrogen']

        user_data = log_transform(user_data, log_columns)
        user_data = sqrt_transform(user_data, sq_columns)
        user_data = power_transform(user_data, pt_columns)

        scaled_data = scaler.transform(user_data)
        user_data = pd.DataFrame(scaled_data, columns=scaler.get_feature_names_out(), index=user_data.index)

        final_features = user_data[['Log_Humidity', 'Log_Rainfall', 'Log_Rainfall_Humidity_Index',
                                    'PT_Potassium', 'Log_Phosphorus', 'Log_NPK_Average',
                                    'Temp_Humididty_Index', 'SQ_Nitrogen', 'Log_PK_Ratio', 'Temperature']]

        prediction = crop_model.predict(final_features)
        predicted_crop = encoder_2.inverse_transform(prediction)[0]

        return render_template('crop-recommendation.html', recommended_crop=predicted_crop)
    except Exception as e:
        return render_template('crop-recommendation.html', error=str(e))


# Plant disease prediction route
@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            output = disease_model(input_tensor)
            probs = F.softmax(output, dim=1)

        pred_class = probs.argmax().item()
        confidence = probs.max().item()

        return render_template('web.html', predicted_class=classes[pred_class])
    except Exception as e:
        return jsonify({"error": "Failed to process image"}), 500


# Weather prediction route
@app.route('/predict_weather', methods=['POST'])
def predict_weather():
    try:
        temperatures = [float(temp.strip()) for temp in request.form.get('temperatures', '').split(',')]
        if len(temperatures) != 10:
            return render_template('xx.html', error_message='Please provide exactly 10 temperature values.')

        temp_input = np.array(temperatures).reshape(1, -1)
        predicted_temp = weather_model.predict(temp_input)[0][0]

        return render_template('xx.html', predicted_temperature=predicted_temp)
    except Exception as e:
        return render_template('xx.html', error_message=str(e))


@app.route('/Dashboard.html')
def dashboard():
    return send_from_directory('templates', 'Dashboard.html')


def generate_agriculture_advice(predicted_temp):
    if predicted_temp < 15:
        return "Low temperature; consider frost protection measures."
    elif 15 <= predicted_temp <= 30:
        return "Optimal temperature for most crops."
    else:
        return "High temperature; ensure adequate irrigation."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)

