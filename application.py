from flask import Flask,render_template,jsonify,request
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

##importing ridge regression model and standard scaler
ridge_model = pickle.load(open("Models/ridge.pkl","rb"))
standard_scaler = pickle.load(open("Models/scaler.pkl","rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature  = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        temp_data = pd.DataFrame(np.array([Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]).reshape(1,-1),
                                       columns=[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region])
        new_scaled_data = standard_scaler.transform(temp_data)
        result = ridge_model.predict(new_scaled_data)

        return render_template('home.html',results = result[0])

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
