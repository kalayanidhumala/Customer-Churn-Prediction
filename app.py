# import numpy as np
# import pandas as pd
# import xgboost
# import pickle
# import sklearn
# import os
# from flask import Flask, render_template, request, send_from_directory

# app = Flask(__name__)

# # model = pickle.load(open('model.pkl', 'rb'))
# # preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
# model_path = os.path.join('Customer_Churn-main', 'model.pkl')
# preprocessor_path = os.path.join('Customer_Churn-main', 'preprocessor.pkl')

# # Load the model and preprocessor from pickle files
# with open(model_path, 'rb') as model_file:
#     model = pickle.load(model_file)

# with open(preprocessor_path, 'rb') as preprocessor_file:
#     preprocessor = pickle.load(preprocessor_file)

# @app.route('/')
# def Home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Gathering inputs
#     account_length = int(request.form.get('account_length'))
#     international_plan = request.form.get('international_plan')
#     vmail_message = int(request.form.get('vmail_message'))
#     day_calls = int(request.form.get('day_calls'))
#     day_charge = float(request.form.get('day_charge'))
#     eve_charge = float(request.form.get('eve_charge'))
#     night_charge = float(request.form.get('night_charge'))
#     international_calls = int(request.form.get('international_calls'))
#     international_charge = float(request.form.get('international_charge'))
#     custServ_calls = int(request.form.get('custServ_calls'))

#     inputs = pd.DataFrame(np.array([account_length, international_plan, vmail_message, day_calls, day_charge, eve_charge, night_charge, international_calls, international_charge, custServ_calls]).reshape(1, -1), columns=['account_length', 'international_plan', 'vmail_message', 'day_calls',
#                                     'day_charge', 'eve_charge', 'night_charge', 'international_calls','international_charge', 
#                                     'custServ_calls'])

#     input_processed = preprocessor.transform(inputs)

#     prediction = model.predict(input_processed)

#     # Generate churn risk scores
#     churn_risk_scores = np.round(model.predict_proba(input_processed)[:, 1] * 100,2)

#     # Churn flag
#     if prediction == 1:
#         prediction = 'YES'
#     else:
#         prediction = 'NO'

#     return render_template('predict.html', prediction=prediction, churn_risk_scores=churn_risk_scores, inputs=request.form)

# if __name__ == '__main__':
#     app.run(debug=True)
import numpy as np
import pandas as pd
import pickle
import os
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the model and preprocessor from pickle files
model_path = os.path.join('Customer_Churn-main', 'model.pkl')
preprocessor_path = os.path.join('Customer_Churn-main', 'preprocessor.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(preprocessor_path, 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Gathering inputs
    account_length = int(request.form.get('account_length'))
    international_plan = request.form.get('international_plan')
    vmail_message = int(request.form.get('vmail_message'))
    day_calls = int(request.form.get('day_calls'))
    day_charge = float(request.form.get('day_charge'))
    eve_charge = float(request.form.get('eve_charge'))
    night_charge = float(request.form.get('night_charge'))
    international_calls = int(request.form.get('international_calls'))
    international_charge = float(request.form.get('international_charge'))
    custServ_calls = int(request.form.get('custServ_calls'))

    inputs = pd.DataFrame(np.array([account_length, international_plan, vmail_message, day_calls, day_charge, eve_charge, night_charge, international_calls, international_charge, custServ_calls]).reshape(1, -1), columns=['account_length', 'international_plan', 'vmail_message', 'day_calls',
                                    'day_charge', 'eve_charge', 'night_charge', 'international_calls','international_charge', 
                                    'custServ_calls'])

    input_processed = preprocessor.transform(inputs)

    prediction = model.predict(input_processed)

    # Generate churn risk scores
    churn_risk_scores = np.round(model.predict_proba(input_processed)[:, 1] * 100,2)

    # Churn flag
    if prediction == 1:
        prediction = 'YES'
    else:
        prediction = 'NO'

    return render_template('predict.html', prediction=prediction, churn_risk_scores=churn_risk_scores, inputs=request.form)

@app.route('/dash')
def dash():
    return render_template('dash.html')

if __name__ == '__main__':
    app.run(debug=True)

