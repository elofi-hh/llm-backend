from flask import Flask, Response, jsonify, request
from flask_cors import CORS, cross_origin
import requests

app = Flask(__name__)
cors = CORS(app, resources={r"/": {"origins": "*"}}, origins=["http://localhost:3000"])
app.config['CORS_HEADERS'] = 'Content-Type'

import random
from transformers import pipeline

model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

@app.route('/', methods=['POST', 'OPTIONS'])
def main():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    data = request.data
    QA_input = {
        'question': f"How many bytes will users on a network for the following business use per 3 seconds: {data}",
        'context': """
            The following is a list of common places and the amount of bytes of data they use per 3 seconds:

            Coffee Shops (e.g., Starbucks, local cafes) use 50000,
            Fast Food Chains (e.g., McDonald's, Taco Bell) use 20000,
            Restaurants (e.g., casual dining, sit-down restaurants) use 30000,
            Retail Stores (e.g., Target, Walmart, Apple Store) use 20000,
            Shopping Malls (common areas and individual stores) use 40000,
            Bookstores (e.g., Barnes & Noble) use 15000,
            Grocery Stores (e.g., Whole Foods, Safeway) use 20000,
            Libraries (public libraries often have free Wi-Fi) use 70000,
            Gyms (e.g., Planet Fitness, 24 Hour Fitness) use 30000,
            Hotels (lobbies and guest areas) use 80000,
            Airports (free or paid Wi-Fi in terminals) use 80000,
            Train/Bus Stations (some transit hubs offer public Wi-Fi) use 50000,
            Public Parks (Wi-Fi zones in major cities) use 40000,
            Hospitals and Clinics (waiting areas) use 90000,
            Banks (some bank branches offer Wi-Fi to customers) use 50000,
            Coworking Spaces (e.g., WeWork, Regus) use 100000
            Movie Theaters (lobbies or waiting areas) use 10000,
            Car Dealerships (customer lounges) use 15000,
            Museums and Galleries (common in major cities) use 15000,
            Community Centers (local government or recreational centers) use 80000,

            IF IT DOES NOT FIT INTO ANY OF THESE CATEGORIES, CHOSE THE AVERAGE BETWEEN MUTIPLE CATEGORIES THAT MAY BE CLOSELY RELATED!
        """
    }
    data = nlp(QA_input)["answer"]

    try:
        data = int(data)
        if data < 10000 or data > 100000:
            data = random.randint(20000, 80000)
    except ValueError:
        print(f"response was {data}")
        data = random.randint(20000, 70000)

    print(data)

    url = 'http://192.168.2.1:8080/onboard'
    myobj = {
        "threshold": data,
        "window": 5
    }

    x = requests.post(url, json = myobj)

    response = Response(status= x.status_code)

    response.headers.add('Access-Control-Allow-Origin', '*')
    
    return response
