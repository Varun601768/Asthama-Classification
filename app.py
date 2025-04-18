from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load the pre-trained model
with open('a_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the pre-trained scaler (used during training)
with open('nr_model.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize Flask app
app = Flask(__name__)

# Define the input features for the model
input_columns = [
    'Age', 'Gender', 'BMI', 'Smoking', 
    'PhysicalActivity', 'DietQuality', 'SleepQuality',
    'FamilyHistoryAsthma', 'LungFunctionFEV1',
     'LungFunctionFVC', 'Wheezing','ShortnessOfBreath'
]
# Route to display the form and handle user input
@app.route("/", methods=["GET", "POST"])
def index():    
    if request.method == "POST":
        # Extract the user input data
        user_input = {
            'Age': int(request.form['Age']),
            'Gender': request.form['Gender'],
            'BMI': float(request.form['BMI']),
            'Smoking': request.form['Smoking'],
            'PhysicalActivity': request.form['PhysicalActivity'],
            'DietQuality': request.form['DietQuality'],
            'SleepQuality': request.form['SleepQuality'],
            'FamilyHistoryAsthma':request.form['FamilyHistoryAsthma'],
            'LungFunctionFEV1':float(request.form['LungFunctionFEV1']),
            'LungFunctionFVC':float(request.form['LungFunctionFVC']),
            'Wheezing':request.form['Wheezing'],
            'ShortnessOfBreath':request.form['ShortnessOfBreath']
        }

        # Convert user input into a DataFrame
        input_df = pd.DataFrame([user_input])

        # Scale the input data using the pre-trained scaler
        input_scaled = scaler.transform(input_df[input_columns])

        # Make prediction
        prediction = model.predict(input_scaled)

        # Display prediction result
        diagnosis = 'Asthma' if prediction[0] == 1 else 'No Asthma'

        return render_template('result.html', diagnosis=diagnosis)
    return render_template("index.html")
@app.route('/about')
def about():
    return render_template('about.html')    
@app.route("/home")
def home():
    return render_template("home.html")


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
