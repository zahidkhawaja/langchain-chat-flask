from flask import Flask, request
from chatbot_qa import new_predict
import os

app = Flask(__name__)

token = os.environ["LCC_TOKEN"]

@app.route("/status")
def status():
    headers = request.headers
    auth = headers.get("X-Api-Key")

    if auth == token:
        return {"success": "Server is running"}, 200
    else:
        return {"error": "Unauthorized"}, 401

@app.route("/answer", methods = ["POST"])
def answer():

    headers = request.headers
    auth = headers.get("X-Api-Key")

    if auth == token:
        # Get the question from the request
        question = request.get_json()["question"]
        history = request.get_json()["history"]
        answer = new_predict(question, history)
        return {"success": answer}, 200
    else:
        return {"error": "Unauthorized"}, 401


if __name__ == '__main__':
    port = os.getenv('PORT',5000)
    app.run(debug=False, host='0.0.0.0', port=port)