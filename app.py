"""
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('rfmodel.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("pathogen_predict.html")


@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = []  # Initialize `selected_symptoms`
    # Get selected symptoms as a list
    for symptom in request.form.getlist('symptoms'):
        selected_symptoms.append(symptom)
    all_symptoms = [ "Chills", "Cough", "Difficulty_breathing", "Sputum_production", "Sore_throat", "Headache", "Runny_nose", "Eye_pain", "Seizures", "Tick_bites", 
                    "Abdominal_pain", "Vomiting", "Diarrhoea", "Blood_in_stool", "Bleeding", "Bruising", "Rash", "Joint_aches", "Muscle_aches", "Dark_urine", "Jaundice" ]
    # Create an array of 0s and 1s based on selected symptoms
    input_features = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]
    
    final = np.array(input_features, dtype=float)
    
    # Assuming `model` is your trained machine learning model
    out = model.predict([final])[0]
    
    return render_template('pathogen_predict.html', pred='Your Pathogen is most likely {}'.format(out))

if __name__ == '__main__':
    app.run(debug=True) 
"""
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('rfmodel.pkl', 'rb'))
selected_symptoms = []  # Initialize `selected_symptoms`

@app.route('/')
def hello_world():
    return render_template("pathogen_predict.html")

@app.route('/predict', methods=['POST'])
def predict():
    global selected_symptoms, prediction_made
    selected_symptoms = request.form.getlist('symptoms')

    if not selected_symptoms:
        out = "No prediction (no symptoms selected)"
    else:
        all_symptoms = ["Chills", "Cough", "Difficulty_breathing", "Sputum_production", "Sore_throat", "Headache", "Runny_nose", "Eye_pain", "Seizures", "Tick_bites", 
                        "Abdominal_pain", "Vomiting", "Diarrhoea", "Blood_in_stool", "Bleeding", "Bruising", "Rash", "Joint_aches", "Muscle_aches", "Dark_urine", "Jaundice"]
        input_features = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]
        final = np.array(input_features, dtype=float)
        out = model.predict([final])[0]
        prediction_made = True  # Set the prediction flag

    return render_template('pathogen_predict.html', pred=' Answer:  {}'.format(out), prediction_made=prediction_made)

if __name__ == '__main__':
    app.run(debug=True)