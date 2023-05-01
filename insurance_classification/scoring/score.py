import json
import joblib
import numpy as np
import os

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'insurance_classification.pkl')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = json.loads(raw_data)['data']
    np_data = np.array(data)
    # Get a prediction from the model
    predictions = model.predict(np_data)
    
    # print the data and predictions (so they'll be logged!)
    log_text = 'Data:' + str(data) + ' - Predictions:' + str(predictions)
    print(log_text)
    
    # Get the corresponding classname for each prediction (0 or 1)
    classnames = ['Yes', 'No']
    predicted_classes = []
    for prediction in predictions:
        predicted_classes.append(classnames[prediction])
    # Return the predictions as JSON
    return json.dumps(predicted_classes)

if __name__ == "__main__":
    # Test scoring
    init()
    test_row = [[2,180,74,24,21,23.9091702,1.488172308,22]]
    prediction = run(test_row)
    print("Test result: ", prediction)