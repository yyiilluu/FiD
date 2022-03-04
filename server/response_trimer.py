import string

TRIM_SET = {", Nate here!"}
import spacy

nlp = spacy.load("en_core_web_sm")

def trim_response(response: str):
    for candidate in TRIM_SET:
        if candidate in response:
            response = response.replace(candidate, ".")

    doc = nlp(response)

    final_sent_texts = [sent.text.strip() for sent in doc.sents]
    last_sent = final_sent_texts[-1]
    word_length = len(last_sent.split())
    if word_length <= 4:
        return " ".join(final_sent_texts[:-1])

    if last_sent[-1] not in string.punctuation:
        last_sent += "."
        final_sent_texts[-1] = last_sent

    return " ".join(final_sent_texts)