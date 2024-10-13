from flask import Flask, Response, request
import requests

app = Flask(__name__)

import random
from transformers import pipeline

model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

@app.route('/', methods=['POST'])
def main():
    data = request.data
    QA_input = {
        'question': f"How many bytes will users on a network for the following business use per 3 seconds: {data}",
        'context': 'Networks that have more uses typically use more bytes of data.  Coffee shops typically use 50000, Department stores usually use 20000.'
    }
    data = nlp(QA_input)["answer"]

    try:
        data = int(data)
    except ValueError:
        print(f"response was {data}")
        data = random.randint(20000, 70000)

    url = 'http://192.168.2.1:8080/onboard'
    myobj = {
        "threshold": data,
        "window": 5
    }

    x = requests.post(url, json = myobj)
    
    return Response(status= x.status_code)
