from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import pytesseract
from werkzeug.utils import secure_filename
import os
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def detect_license_plate(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    license_plate = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            license_plate = image[y:y+h, x:x+w]
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            break
    
    if license_plate is None:
        return "License plate not detected", None

    license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(license_plate_gray, config='--psm 8').strip()

    # Convert image to base64 string
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    img_data = f"data:image/jpeg;base64,{img_str}"
    return text, img_data

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            text, img_data = detect_license_plate(filepath)
            return render_template('result.html', text=text, input_image=filepath, output_image=img_data)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def send_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
