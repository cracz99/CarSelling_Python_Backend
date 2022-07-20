from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
})
import os
port = int(os.environ.get("PORT", 8000))
import pickle
import numpy as np
import pandas as pd

@app.route('/predict_loan_eligibility', methods=['POST'])
def predictLoanEligibility():
    data=request.get_json(force=True)
    KeysExpected = ["ApplicantIncome", "CoapplicantIncome",
                    "LoanAmount", "Credit_History", "Gender", "Married", "Dependents", "Education", "Property_Area"]
 
    #Check to ensure all the fields are available from the input
    for key in KeysExpected:
        if key not in data:
            return jsonify({'success': False, "Message": key+" field is missing",key:key})
 
    #convert data to dataframe
    df=pd.DataFrame.from_dict(data)

    try:
        loaded_model = pickle.load(open("./Model/randomForest.sav", 'rb'))
        #Get the first row of dataframe
        parameters = df.iloc[0]
        #Predict if eligible or not
        prediction = loaded_model.predict([parameters])
        #Get the score out of 1
        score= loaded_model.predict_proba([parameters])
        
        eligible=False
        if(prediction[0] == 1):
            eligible=True

        return jsonify({'success':True,'eligible': eligible, 'creditScore': score[0][1]})
    
    except ValueError:
        return jsonify({'success': False, 'Message': "Invalid field data type"})

    
@app.errorhandler(404)
def page_not_found(error):
    return jsonify({'success': False,'Message':'Not found'})
    
if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
  
    app.run(debug=True, threaded=True, host='0.0.0.0', port=port)