import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
sc = pickle.load(open('sc.pkl', 'rb'))
model = pickle.load(open('classifier.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
# def predict():

#     float_feature = [float(x) for x in request.form.values()]
#     print(float_feature)
#     final_feature = [np.array(float_feature)]
#     print(final_feature)
#     pred = model.predict(sc.transform(final_feature))
#     return render_template('result.html', prediction = pred)
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        float_feature = np.array([preg, glucose, bp, st, insulin, bmi, dpf, age])
        print(float_feature)
        final_feature = [np.array(float_feature)]
        print(final_feature)
        pred = model.predict(sc.transform(final_feature))
        return render_template('result.html', prediction = pred)

if __name__ == "__main__":
    app.run(debug=True)