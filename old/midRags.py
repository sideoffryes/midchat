from transformers import pipeline
import re

pipe = pipeline("question-answering", model="distilbert-base-cased", framework="pt", tokenizer="distilbert-base-cased")

def clean(text):
    # get rid of all newline chars
    text = re.sub(r'\n', ' ', text)
    
    # get rid of all tab chars
    text = re.sub(r'\t', ' ', text)
    
    # get rid of many consecutive spaces
    text = re.sub(r" {2,}", " ", text)

    text = text.strip()

    return text

def load_text(path: str, max_len=500) -> list:
    with open(path, "r") as file:
        text = file.read()

        text = clean(text).split(" ")

        chunks = [" ".join(text[i:i+max_len]) for i in range(0, len(text), max_len)]

        return chunks
    
def test_context(chunks: list, question: str) -> str:
    best_answer = ""
    best_score = 0

    for c in chunks:
        result = pipe(question=question, context=c)
        if result["score"] > best_score:
            best_answer = result["answer"]
            best_score = result["score"]

    print(f"Best score: {best_score}")

    return best_answer

if __name__ == "__main__":
    chunks = load_text("./midregs.txt")
    question = input("Enter prompt> ").strip()

    while question != "quit":
        print(test_context(chunks, question))
        question = input("Enter prompt> ").strip()