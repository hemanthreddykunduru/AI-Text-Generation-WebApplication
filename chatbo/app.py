from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load the text generation pipeline
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')

@app.route('/', methods=['GET', 'POST'])
def home():
    generated_text = ""
    if request.method == 'POST':
        prompt = request.form['prompt']
        if prompt:
            # Generate text based on the user's input
            res = generator(prompt, max_length=5, do_sample=True, temperature=0.9)
            generated_text = res[0]['generated_text']

    return render_template('index.html', generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)
