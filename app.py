from flask import Flask, request, jsonify
import spacy
import pdfplumber

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())

@app.route("/rank", methods=["POST"])
def rank_resume():
    file = request.files["resume"]
    job_description = request.form["job"]
    
    resume_text = extract_text_from_pdf(file)
    resume_doc = nlp(resume_text)
    job_doc = nlp(job_description)

    similarity = resume_doc.similarity(job_doc)
    score = round(similarity * 100, 2)

    return jsonify({"score": score})

if __name__ == "__main__":
    app.run(debug=True)
