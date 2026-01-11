from flask import Flask, request, jsonify
from rag_pipeline import ask_finance_question

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return {"message": "Personal Finance Advisor API is running"}

@app.route("/ask-test", methods=["GET"])
def ask_test():
    answer = ask_finance_question("How should a student save money?")
    return {"answer": answer}

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    answer = ask_finance_question(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
