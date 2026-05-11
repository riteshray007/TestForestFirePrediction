from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

elasticnet_model = pickle.load(open('models/elasticnetcv.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
      return render_template('index.html')


@app.route('/predictdata' , methods=['get','post'])
def predict_datapoint():
      if request.method=='POST':
            day=float(request.form.get('day'))
            month=float(request.form.get('month'))
            Temperature=float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request. form.get('Ws'))
            Rain = float(request. form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            DC = float(request.form.get('DC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))
            
            new_data_scaled = standard_scaler.transform([[day,month,Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,Classes,Region]])
            print(new_data_scaled)
            result = elasticnet_model.predict(new_data_scaled)
            print(result)
            return render_template('home.html' , result=result)
            
      else:
            return render_template('home.html')



if __name__ == "__main__":
      app.run(host="0.0.0.0")
