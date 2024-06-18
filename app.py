from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

dic = {
    0: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    1: 'Corn_(maize)___Common_rust_',
    2: 'Corn_(maize)___Northern_Leaf_Blight',
    3: 'Corn_(maize)___healthy',
    4: 'Potato___Early_blight',
    5: 'Potato___Late_blight',
    6: 'Potato___healthy',
    7: 'Tomato___Bacterial_spot',
    8: 'Tomato___Early_blight',
    9: 'Tomato___Late_blight',
    10: 'Tomato___Leaf_Mold',
    11: 'Tomato___Septoria_leaf_spot',
    12: 'Tomato___Spider_mites Two-spotted_spider_mite',
    13: 'Tomato___Target_Spot',
    14: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    15: 'Tomato___Tomato_mosaic_virus',
    16: 'Tomato___healthy',
}

model = load_model('sani_massive.h5')

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i) / 255.0
    i = np.expand_dims(i, axis=0)
    p = model.predict(i)
    return dic[np.argmax(p)]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("classification.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        
        if img and img.filename != '':
            img_path = os.path.join('static', img.filename)
            img.save(img_path)
            p = predict_label(img_path)
            return render_template("classification.html", prediction=p, img_path=img_path)
        else:
            return render_template("classification.html", prediction="No file uploaded.", img_path=None)

if __name__ == '__main__':
    app.run(debug=True)
