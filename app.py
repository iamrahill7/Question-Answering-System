from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)
qa_pipeline = pipeline(
    'question-answering', model='deepset/roberta-base-squad2')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/answer', methods=['POST'])
def answer():
    question = request.form['question']
    context = request.form['context']
    answer = qa_pipeline(question=question, context=context)
    return render_template('index.html', question=question, context=context, answer=answer['answer'])


if __name__ == '__main__':
    app.run(debug=True)
