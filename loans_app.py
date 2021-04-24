

# import Flask and jsonify
import flask
from flask import render_template, request, jsonify,Flask
# import Resource, Api and reqparser
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy as np
import traceback
import pickle


app = Flask(__name__)
api = Api(app)


# --------------------------------- #
# INFORMATION FOR MODEL DEVELOPMENT #
# create new column given df
def totalIncome(dataset):
    dataset['TotalIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
    return dataset
# allow function transformation on df column
class DataframeFunctionTransformer():
    def __init__(self, func):
        self.func = func
    def transform(self, input_df, **transform_params):
        return self.func(input_df)
    def fit(self, X, y=None, **fit_params):
        return self
# log transform given columns
def LogTransform(dataset):
    dataset = dataset.assign(log_TotalIncome = np.log1p(dataset['TotalIncome']))
    dataset = dataset.assign(log_LoanAmount = np.log1p(dataset['LoanAmount']))
    dataset = dataset.drop(columns=['TotalIncome','LoanAmount'])
#    dataset['TotalIncome'].assign() = dataset['TotalIncome'].apply(np.log)
#    dataset['LoanAmount'] = dataset['LoanAmount'].apply(np.log)
    return dataset
# don't forget ToDenseTransformer after one hot encoder
class ToDenseTransformer():
    # here you define the operation it should perform
    def transform(self, X, y=None, **fit_params):
        return X.todense()
    # just return self
    def fit(self, X, y=None, **fit_params):
        return self
# select specific columns to perform pipeline onto
class SelectColumnsTransformer():
    def __init__(self, columns=None):
        self.columns = columns
    def transform(self, X, **transform_params):
        cpy_df = X[self.columns].copy()
        return cpy_df
    def fit(self, X, y=None, **fit_params):
        return self
# INFORMATION FOR MODEL DEVELOPMENT #
# --------------------------------- #


# import model
with open("xtrain_logregmodel.sav", "rb" ) as f:
    regressor = pickle.load (f)

@app.route('/')
def instructions():
    return 'Welcome to the Flask App for Loan Predictions! POST your json parameters to get specific predictions over at '/predict''

# assign endpoint
@app.rout('/predict', methods=['POST','GET'])
def predict():

   if flask.request.method == 'GET':
       return "Prediction page. Try using POST with params to get specific prediction."

   if flask.request.method == 'POST':
       try:
           json_ = request.json
           print(json_)
           query_ = pd.get_dummies(pd.DataFrame(json_))
           prediction = list(regressor.predict(query))

           return jsonify({
               "loan status prediction":str(prediction)
           })

       except:
           return jsonify({
               "trace": traceback.format_exc()
               })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5555)


