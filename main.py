from flask import Flask, request

app = Flask(__name__)

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
    res = nlp(QA_input)
    return res["answer"]
