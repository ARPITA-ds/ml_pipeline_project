from flask import Flask
from mlpipeline.logger import logging
from mlpipeline.exception import CustomException
import os,sys

app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])
def index():
    try:
        raise Exception("we are testing our custom file")
        return "Welcome to ML project session"
    except Exception as e:
        abc= CustomException(e,sys)
        logging.info(abc.error_message)
        return "Welcome to ML project session"
if __name__ == '__main__':
    app.run(debug=True)