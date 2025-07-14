from fastapi import FastAPI, Query
from rag_chatbot import answer_question 

app = FastAPI()

@app.get("/ask")
def ask(question: str = Query(..., description="Pytanie do systemu")):
    answer = answer_question(question)
    return {"odpowiedź": answer}
