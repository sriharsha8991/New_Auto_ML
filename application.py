import os
import sys
from flask import Flask, request, render_template
import pandas as pd
import pandas_profiling
import io
from src.logger import logging
from src.components.Automl import Automl
from flaml import AutoML

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST','GET'])
def upload_file():
    uploaded_file = request.files['file']
    file_ext = uploaded_file.filename.split('.')[-1].lower()
    if file_ext == 'csv':
       df = pd.read_csv(uploaded_file)
    elif file_ext in ['xls', 'xlsx']:
        logging.info("Making excel doc into csv...")
        df = pd.read_excel(uploaded_file)
        # convert the excel file to csv format
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        df = pd.read_csv(io.StringIO(csv_buffer.getvalue()))

    else:
        return 'Invalid file format. Only CSV and Excel files are supported.'
        logging.info("Invalid Format")



    profile = pandas_profiling.ProfileReport(df)
    output = request.form.get('output')
    if output == 'csv':
        response = pd.DataFrame.to_csv(df, index=False)
        logging.info("Exploration is completed")
        return response
    elif output == 'automl':
        obj = Automl()
        a,b,c,d = obj.automate(df)
        kk = pd.DataFrame({
            "Best ML learner": [a],
            'Best hyperparmeter config':[b], 
            'Best accuracy on validation data':[c],
            'Training duration of best run': [d]
    
        })
        return render_template("table.html",results=kk.to_html())
    else:
        profile_html = profile.to_html()
        return profile_html
    

# @app.route("/automl", method=['POST'])
# def automl():
#     file_ext = uploaded_file.filename.split('.')[-1].lower()
#     if file_ext == 'csv':
#         df = pd.read_csv(uploaded_file)
#     elif file_ext in ['xls', 'xlsx']:
#         logging.info("Making excel doc into csv...")
#         df = pd.read_excel(uploaded_file)
#         # convert the excel file to csv format
#         csv_buffer = io.StringIO()
#         df.to_csv(csv_buffer, index=False)
#         df = pd.read_csv(io.StringIO(csv_buffer.getvalue()))
#     else:
#         return 'Invalid file format. Only CSV and Excel files are supported.'
#         logging.info("Invalid Format")
    
#     obj = Automl()
#     a,b,c,d = obj.automate(df1)
#     return (('Best ML leaner:',a),
#            ('Best hyperparmeter config:', b),
#            ('Best accuracy on validation data: ',c),
#            ('Training duration of best run: {}s'.format(d)))


    



if __name__ == '__main__':
    app.run(debug=True)
