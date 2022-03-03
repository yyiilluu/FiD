from fastapi import FastAPI

from inference import inference
from search import search

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/test")
async def root(username, ticket_subject, ticket_body):
    ticket_subject = ""
    ticket_body = ""
    username = ""
    ctxs = search(ticket_subject=ticket_subject, ticket_body=ticket_body)
    question = f""
    input_example = constrcut_example(question, ctxs)
    response = inference(input_example, opt)



    return {"generated_response": response, "ctxs": ctxs}