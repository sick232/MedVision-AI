from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# --- Configuration ---
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Model Loading ---
# Object Detection Models
MODELS = {
    'liver': {'path': 'models/liver_tumour.pt', 'classes': ["liver", "tumor"]},
    'brain_mri': {'path': 'models/brain_mri.pt', 'classes': ["tumour", "eye"]},
    'kidney': {'path': 'models/kidney.pt', 'classes': ["stone"]},
    'lung': {'path': 'models/lung cancer.pt', 'classes': ["Tumour"]}
}
models = {name: YOLO(config['path']) for name, config in MODELS.items()}

# Generative AI Model (Gemini)
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    print("Warning: GEMINI_API_KEY not found in .env file. AI assistant will not work.")
    gemini_model = None
else:
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")


# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def find_initial_positions(image_path, model_name):
    model = models.get(model_name) or models[list(models.keys())[0]]
    class_names = MODELS.get(model_name, {}).get('classes', [])
    results = model.predict(source=image_path, conf=0.40)

    detections = []
    for detection in results[0].boxes:
        x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])
        class_label = class_names[int(detection.cls[0])] if int(detection.cls[0]) < len(class_names) else "Unknown"
        detections.append({'class': class_label.lower(), 'bbox': [x_min, y_min, x_max, y_max]})
    return detections


# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if not file or not file.filename: return jsonify({'error': 'No selected file'}), 400

    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        selected_model = request.form.get('model', 'liver')

        try:
            detections = find_initial_positions(filepath, selected_model)
            primary_target = next((d for d in detections if any(k in d['class'] for k in ["tumor", "tumour", "stone"])),
                                  None)
            if not primary_target: return jsonify({'error': 'No primary target detected.'}), 404

            return jsonify({
                'original_image': f'/{filepath}',
                'detections': detections,
                'target_class': primary_target['class']
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'File type not allowed'}), 400


@app.route('/get_gemini_advice', methods=['POST'])
def get_gemini_advice():
    if not gemini_model:
        return jsonify({'advice': "AI Assistant is offline. Proceed with caution."})

    data = request.json
    context = (
        f"You are a senior surgical assistant AI. Guide a surgeon in a simulated procedure. "
        f"The primary target is a {data.get('target_class', 'lesion')}. "
        f"The last maneuver was a {data.get('distance', 'N/A')}mm repositioning, classified as a '{data.get('severity', 'N/A')}' move. "
        f"A 'no-fly zone' intersection was {data.get('nfz_breach', 'not')} detected. "
        f"Provide a concise, actionable instruction for the next immediate step. Be direct and clear."
    )

    try:
        response = gemini_model.generate_content(context)
        cleaned_advice = response.text.replace('*', '').strip()
        return jsonify({'advice': cleaned_advice})
    except Exception as e:
        return jsonify({'advice': f"Error from AI assistant: {str(e)}"})


@app.route('/get_procedure_summary', methods=['POST'])
def get_procedure_summary():
    if not gemini_model:
        return jsonify({'summary': "AI Assistant is offline. Unable to generate a post-procedure summary."})

    data = request.json
    context = (
        f"You are a senior surgeon AI providing a post-procedure summary. The procedure targeted a {data.get('target_class', 'lesion')}. "
        f"The procedure involved {data.get('total_moves', 'a number of')} maneuvers. "
        f"The severity breakdown was: {data.get('severity_breakdown', 'N/A')}. "
        f"There were {data.get('nfz_breaches', 0)} breaches of critical 'no-fly zones'. "
        f"Provide a brief, professional summary of the procedure's execution and overall performance. Comment on the efficiency and safety demonstrated."
    )

    try:
        response = gemini_model.generate_content(context)
        cleaned_summary = response.text.replace('*', '').strip()
        return jsonify({'summary': cleaned_summary})
    except Exception as e:
        return jsonify({'summary': f"Error generating summary: {str(e)}"})


if __name__ == '__main__':
    app.run(debug=True)