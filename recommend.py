import pickle as pkl
import json
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

   
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data=data['drug_composition']
    data=np.array(data)
    predicted_values = model.predict(data)
    model=pkl.load(open('model.pkl','rb'))
    return jsonify({"Dissolution_av":predicted_values[0, 0]},
                   {"Dissolution_min":predicted_values[0, 1]},
                   {"Residual_solvent":predicted_values[0, 2]},
                   {"Impurities_total":predicted_values[0, 3]},
                   {"Impurity_o":predicted_values[0, 4]},
                   {"Impurity_l":predicted_values[0, 5]})

if __name__ == "__main__":
    app.run()
