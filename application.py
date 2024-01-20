from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
application = Flask(__name__)
app = application
# Import ridge regression model and standard scaler pickle
logistic_model = pickle.load(open('models/model.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
@app.route("/", methods=["GET", "POST"])
def predict_datapoint():
    result=""
    if request.method == 'POST':
        Pregnancies= int(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))
        
        new_data_scaled = standard_scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict = logistic_model.predict(new_data_scaled)
        if predict[0] ==1:
         result= 'Diabetic'
        else:
            result= 'Non-Diabetic' 

        return render_template('prediction.html', result=result)

    else:
        return render_template('home.html')

    

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5003)
