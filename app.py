from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    Origin = request.form['Origin']
    Source = request.form['Source']
    Do_not_Email = request.form['Do_not_Email']
    Total_Visits = int(request.form['Total_Visits'])
    Time_spend_on_site = float(request.form['Time_spend_on_site'])
    Page_Viewed = float(request.form['Page_Viewed'])
    LA = request.form['LA']
    Region = request.form['Region']
    Job = request.form['Job']
    Free_copy = request.form['Free_copy']

    # Create DataFrame for categorical data
    categorical_data = pd.DataFrame({
        'Origin': [Origin],
        'Source': [Source],
        'LA': [LA],
        'Region': [Region],
        'Job': [Job]
    })

    # One-hot encoding of categorical variables
    categorical_data = pd.get_dummies(categorical_data)

    # Ensure all expected columns are present
    expected_columns = ['Origin_API', 'Origin_Landing Page Submission', 'Origin_Lead Add Form',
                        'Origin_Lead Import', 'Origin_Quick Add Form', 'Source_Direct Traffic',
                        'Source_Online Chat', 'Source_Referral Sites', 'Source_Search Source',
                        'Source_Social Media', 'Source_Welingak Website', 'LA_Converted to Lead',
                        'LA_Email Bounced', 'LA_Email Link Clicked', 'LA_Email Opened',
                        'LA_Form Submitted on Website', 'LA_Had a Phone Conversation',
                        'LA_Olark Chat Conversation', 'LA_Page Visited on Website', 'LA_SMS Sent',
                        'LA_Unreachable','LA_Unsubscribed','Region_Africa', 'Region_America', 'Region_Asia',
                        'Region_Australia', 'Region_Europe', 'Region_Middle_East', 'Job_Businessman',
                        'Job_Housewife', 'Job_Student', 'Job_Unemployed','Job_Working Professional']

    # Add missing columns as 0 if they don't exist in the data
    for col in expected_columns:
        if col not in categorical_data.columns:
            categorical_data[col] = 0

    # Align categorical data to expected columns
    categorical_data = categorical_data[expected_columns]

    # Create DataFrame for numerical data
    numerical_data = pd.DataFrame({
        'Do_not_Email': [Do_not_Email],
        'Total_Visits': [Total_Visits],
        'Time_spend_on_site': [Time_spend_on_site],
        'Page_Viewed': [Page_Viewed],
        'Free_copy': [Free_copy]
    })

    # Concatenate numerical and categorical data
    final_data = pd.concat([numerical_data, categorical_data], axis=1)

    # Feature scaling only on numerical features
    scaler = StandardScaler()
    final_data[['Total_Visits', 'Time_spend_on_site', 'Page_Viewed']] = scaler.fit_transform(
        final_data[['Total_Visits', 'Time_spend_on_site', 'Page_Viewed']]
    )

    # Make prediction using the trained model
    prediction = model.predict(final_data)[0]

    # Render the result page with the prediction
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
