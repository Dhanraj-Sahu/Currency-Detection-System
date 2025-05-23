import os
import numpy as np
import cv2
import mahotas
import pickle
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

# Flask app configuration
app = Flask(__name__)
app.secret_key = "currency_detection_secret_key"
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure paths
BOVW = "model/bovw_codebook_600.pickle"
MODEL = 'model/rfclassifier_600.sav'
IMG_SIZE = 320

# Class-label dictionary
label = {0: "10", 1: "20", 2: "50", 3: "100", 4: "200", 5: "500", 6: "2000"}

# Load the model and BOVW codebook
try:
    loaded_model = joblib.load(MODEL)
    pickle_in = open(BOVW, "rb")
    dictionary = pickle.load(pickle_in)

    # Initialize SIFT BOW image descriptor extractor
    sift2 = cv2.xfeatures2d.SIFT_create()
    bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
    bowDiction.setVocabulary(dictionary)
    print("Model and dictionary loaded successfully")
except Exception as e:
    print(f"Error loading model or dictionary: {e}")

# Feature extraction functions
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def fd_histogram(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bins = 8
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def feature_extract(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    feature = bowDiction.compute(gray, sift.detect(gray))
    return feature.squeeze()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_currency(image_path):
    try:
        # Read and resize image
        image = cv2.imread(image_path)
        (height, width, channel) = image.shape
        resize_ratio = 1.0 * (IMG_SIZE / max(width, height))
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        input_image = cv2.resize(image, target_size)
        
        # Save resized image for display
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], "resized_" + os.path.basename(image_path))
        cv2.imwrite(result_path, input_image)
        
        # Extract features
        Hu = fd_hu_moments(input_image)
        Harl = fd_haralick(input_image)
        Hist = fd_histogram(input_image)
        Bovw = feature_extract(input_image)
        
        # Generate a feature vector by combining all features
        mfeature = np.hstack([Hu, Harl, Hist, Bovw])
        
        # Predict the output using trained model
        output = loaded_model.predict(mfeature.reshape((1, -1)))
        proba = loaded_model.predict_proba(mfeature.reshape((1, -1)))[0]
        
        # Get the prediction and confidence
        prediction = label[output[0]]
        confidence = round(proba[output[0]] * 100, 2)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'result_image': "resized_" + os.path.basename(image_path)
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the image and get prediction
        result = predict_currency(file_path)
        
        if 'error' in result:
            flash(f"Error in processing: {result['error']}")
            return redirect(url_for('index'))
        
        # Return result template with predictions
        return render_template('result.html', 
                              prediction=result['prediction'],
                              confidence=result['confidence'],
                              image=filename,
                              result_image=result['result_image'])
    else:
        flash('File type not allowed. Please upload jpg, jpeg or png files only.')
        return redirect(url_for('index'))

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True) 