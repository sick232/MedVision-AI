from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import torch
# Patch torch.load to disable weights_only for YOLO models (PyTorch 2.6+ compatibility)
# This is safe since we trust the model files
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv


from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter


load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# --- Configuration ---
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODELS = {
    'liver': {'path': 'models/liver_tumour.pt', 'classes': ["liver", "tumor"]},
    'brain_mri': {'path': 'models/brain_mri.pt', 'classes': ["tumour", "eye"]},
    'eye': {'path': 'models/eye.pt', 'classes': ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]},
    'kidney': {'path': 'models/kidney.pt', 'classes': ["stone"]},
    'lung': {'path': 'models/lung cancer.pt', 'classes': ["Tumour"]}
}
# Load models with error handling for PyTorch 2.6+ compatibility
models = {}
for name, config in MODELS.items():
    try:
        models[name] = YOLO(config['path'])
        print(f"Successfully loaded {name} model")
    except Exception as e:
        print(f"Warning: Could not load {name} model: {e}")
        # Create a dummy model or skip
        models[name] = None


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


def process_image(image_path, model_name=None):

    if model_name is None:
        results_all = {}
        for name, model in models.items():
            if model is None:
                print(f"Skipping {name} model - not loaded")
                continue
            results = model.predict(source=image_path, conf=0.25)
            results_all[name] = {
                'results': results[0],
                'classes': MODELS[name]['classes']
            }

        return process_multiple_model_results(image_path, results_all)

    # If a specific model is chosen
    model = models[model_name]
    if model is None:
        raise ValueError(f"Model {model_name} is not available")
    class_names = MODELS[model_name]['classes']

    # Make prediction
    results = model.predict(source=image_path, conf=0.25)
    result = results[0]

    # Load and process image
    img = cv2.imread(image_path)

    # Store detections info
    detections = []

    # Draw boxes and apply pseudo-coloring
    for detection in result.boxes:
        x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])
        confidence = float(detection.conf[0])
        class_id = int(detection.cls[0])  # Extract class index

        # Get class name
        class_label = class_names[class_id] if class_id < len(class_names) else "Unknown"

        # Store detection info
        detections.append({
            'class': class_label,
            'confidence': confidence,
            'bbox': [x_min, y_min, x_max, y_max]
        })

        # Define color
        color = (0, 255, 0)  # Default green
        if model_name == 'liver':
            color = (0, 255, 0) if class_label == "liver" else (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)


        text = f"{class_label.capitalize()} ({confidence:.2f})"
        cv2.putText(img, text, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Pseudo-coloring for specific cases (customize as needed)
        if class_label in ["tumor", "tumour", "cancer", "nodule", "stone"]:
            detected_area = img[y_min:y_max, x_min:x_max]
            gray_area = cv2.cvtColor(detected_area, cv2.COLOR_BGR2GRAY)
            pseudo_colored_area = cv2.applyColorMap(gray_area, cv2.COLORMAP_JET)
            img[y_min:y_max, x_min:x_max] = cv2.addWeighted(
                detected_area, 0.5, pseudo_colored_area, 0.5, 0
            )

    # Save processed image
    output_path = os.path.join(app.config['UPLOAD_FOLDER'],
                               'processed_' + os.path.basename(image_path))
    cv2.imwrite(output_path, img)

    return output_path, detections


def process_multiple_model_results(image_path, results_all):
    # Load original image
    img = cv2.imread(image_path)

    # Collect all detections
    all_detections = {}

    # Process results from each model
    for model_name, model_results in results_all.items():
        result = model_results['results']
        class_names = model_results['classes']

        model_detections = []

        for detection in result.boxes:
            x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])
            confidence = float(detection.conf[0])
            class_id = int(detection.cls[0])

            # Get class name
            class_label = class_names[class_id] if class_id < len(class_names) else "Unknown"

            # Store detection info
            model_detections.append({
                'class': class_label,
                'confidence': confidence,
                'bbox': [x_min, y_min, x_max, y_max]
            })

            # Define color (you might want to customize this)
            color = (0, 0, 255)  # Red for multi-model detections

            # Draw bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # Add text label
            text = f"{model_name.capitalize()}: {class_label.capitalize()} ({confidence:.2f})"
            cv2.putText(img, text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Store detections for this model
        all_detections[model_name] = model_detections

    # Save processed image
    output_path = os.path.join(app.config['UPLOAD_FOLDER'],
                               'processed_multi_' + os.path.basename(image_path))
    cv2.imwrite(output_path, img)

    return output_path, all_detections


def generate_response_with_image(image_path, prompt):
    if not os.path.exists(image_path):
        return "Error: Image file not found"
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
    except Exception as e:
        return f"Error reading the image: {e}"

    contents = [
        {"mime_type": "image/jpeg", "data": image_data},
        prompt
    ]

    try:
        response = gemini_model.generate_content(contents=contents)
        response.resolve()

        # Clean the response by removing unwanted characters or asterisks
        cleaned_response = response.text.replace('*', '').strip()

        return cleaned_response
    except Exception as e:
        return f"Error Generating Response: {e}"

def create_pdf_report(pdf_filepath, original_image_path, processed_image_path, detections, gemini_response):
    doc = SimpleDocTemplate(pdf_filepath, pagesize=letter,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title = Paragraph("Medical Image Detection Report", styles['h1'])
    story.append(title)
    story.append(Spacer(1, 0.25 * inch))

    # --- Add Images ---
    # Set a max width for the images to ensure they fit on the page
    max_img_width = 6 * inch

    # Original Image
    story.append(Paragraph("Original Image:", styles['h2']))
    img1 = Image(original_image_path)
    img_width, img_height = img1.drawWidth, img1.drawHeight
    img1.drawWidth = min(img_width, max_img_width)
    img1.drawHeight = img_height * (img1.drawWidth / img_width)  # Maintain aspect ratio
    story.append(img1)
    story.append(Spacer(1, 0.25 * inch))

    # Processed Image
    story.append(Paragraph("Processed Image:", styles['h2']))
    img2 = Image(processed_image_path)
    img_width, img_height = img2.drawWidth, img2.drawHeight
    img2.drawWidth = min(img_width, max_img_width)
    img2.drawHeight = img_height * (img2.drawWidth / img_width)  # Maintain aspect ratio
    story.append(img2)
    story.append(Spacer(1, 0.25 * inch))

    # --- Add Detections (with proper formatting) ---
    story.append(Paragraph("Quantitative Findings:", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))

    if isinstance(detections, dict):
        has_findings = any(model_detections for model_detections in detections.values())
        if not has_findings:
            story.append(Paragraph("No significant findings were detected by the AI models.", styles['BodyText']))
        else:
            for model_name, model_detections in detections.items():
                if not model_detections: continue
                story.append(Paragraph(f"<b>Model: {model_name.replace('_', ' ').capitalize()}</b>", styles['h3']))
                for detection in model_detections:
                    detection_text = f"&nbsp;&nbsp;•&nbsp;<b>{detection['class'].capitalize()}</b> detected with <b>{detection['confidence'] * 100:.2f}%</b> confidence."
                    story.append(Paragraph(detection_text, styles['BodyText']))
                story.append(Spacer(1, 0.1 * inch))
    else:
        if not detections:
            story.append(Paragraph("No significant findings were detected.", styles['BodyText']))
        else:
            for detection in detections:
                detection_text = f"•&nbsp;<b>{detection['class'].capitalize()}</b> detected with <b>{detection['confidence'] * 100:.2f}%</b> confidence."
                story.append(Paragraph(detection_text, styles['BodyText']))

    story.append(Spacer(1, 0.25 * inch))

    # --- Add Gemini Response (with text wrapping) ---
    story.append(Paragraph("Clinical AI Insights:", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))
    # Replace newlines with <br/> for proper paragraph breaks in ReportLab
    formatted_response = gemini_response.replace('\n', '<br/>')
    story.append(Paragraph(formatted_response, styles['BodyText']))

    # Build the PDF
    doc.build(story)


# --- SIMULATION PAGE LOGIC (FROM YOUR ORIGINAL simulate.py) ---
def find_initial_positions_for_simulation(image_path, model_name):
    model = models.get(model_name)
    if model is None:
        # Try to find any available model
        for name, m in models.items():
            if m is not None:
                model = m
                model_name = name
                break
        if model is None:
            raise ValueError("No models are available")
    
    class_names = MODELS.get(model_name, {}).get('classes', [])
    results = model.predict(source=image_path, conf=0.40)

    detections = []
    for detection in results[0].boxes:
        x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])
        class_label = class_names[int(detection.cls[0])] if int(detection.cls[0]) < len(class_names) else "Unknown"
        detections.append({'class': class_label.lower(), 'bbox': [x_min, y_min, x_max, y_max]})
    return detections


# --- FLASK ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    selected_model = request.form.get('model', None)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            processed_path, detections = process_image(filepath, selected_model)

            prompt_text = "Describe the medical findings in the image with a professional perspective, including possible next steps in diagnosis and treatment, such as examinations, tests, and medications. If no significant findings are present, please confirm that. Provide insights as a doctor with 50 years of experience, considering both traditional and modern approaches, in no more than 10 lines."
            response_text = generate_response_with_image(processed_path, prompt_text)

            pdf_filename = f"report_{filename.rsplit('.', 1)[0]}.pdf"
            pdf_filepath = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)

            # --- UPDATED FUNCTION CALL ---
            create_pdf_report(pdf_filepath, filepath, processed_path, detections, response_text)

            return jsonify({
                'message': 'Detection completed',
                'original_image': f'/static/uploads/{filename}',
                'processed_image': f'/static/uploads/{os.path.basename(processed_path)}',
                'detections': detections,
                'gemini_response': response_text,
                'pdf_report': f'/static/uploads/{pdf_filename}'
            })
        except Exception as e:
            # Log the full exception for debugging
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/simulate')
def simulate_home():
    return render_template('simulate.html')


@app.route('/detect_for_simulation', methods=['POST'])
def detect_for_simulation():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if not file or not file.filename: return jsonify({'error': 'No selected file'}), 400

    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        selected_model = request.form.get('model', 'liver')

        try:
            detections = find_initial_positions_for_simulation(filepath, selected_model)
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
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)