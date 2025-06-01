from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

# To run python -m uvicorn ml_api:app

app = FastAPI()

class ModelInput(BaseModel): # For API knowing its Type [cause of json]
    pregnancies: int
    glucose: int
    blood_pressure: int
    skin_thickness: int
    insulin: int
    bmi: float
    diabetes_pedigree_function: float
    age: int


# loading the saved model
model = pickle.load(open('Model.sav','rb'))


@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters: ModelInput):

    input_data = input_parameters.json()   # input data is converted to json
    input_dict = json.loads(input_data) # converting Json to Dictonaries

    pregnancies = input_dict['pregnancies']
    glucose = input_dict['glucose']
    blood_pressure = input_dict['blood_pressure']
    skin_thickness = input_dict['skin_thickness']
    insulin = input_dict['insulin']
    bmi = input_dict['bmi']
    diabetes_pedigree = input_dict['diabetes_pedigree_function']
    age = input_dict['age']

    input_list = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]

    prediction = model.predict([input_list])

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


