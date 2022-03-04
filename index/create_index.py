import json
from typing import Dict, List

from elasticsearch import Elasticsearch


def index(index_ticket: bool, index_kas: bool):
    es = Elasticsearch("http://localhost:9200")

    if index_ticket:
        ticket_index_name = "tickets"

        # load data
        with open("index_data/index_tickets_traindata.json", "r") as f:
            tickets_and_responses = json.load(f)

        # index tickets
        if not es.indices.exists(index=ticket_index_name):
            es.indices.create(index=ticket_index_name)
        for elem in tickets_and_responses:
            es.index(index=ticket_index_name, body=elem)

    if index_kas:
        # index knowledge articles
        kas_index_name = "knowledge_articles"
        with open("index_data/indexable_paragraphs_from_train.json", "r") as f:
            kas = json.load(f)
        if not es.indices.exists(index=kas_index_name):
            es.indices.create(index=kas_index_name)
        for elem in kas:
            ka_body = elem["paragraph"]
            ka_title = elem["document_title"]
            ka_id = elem["id"]
            body = {
                "paragraph": ka_title + "\n\n" + ka_body,
                "id": ka_id,
                "document_title": ka_title,
            }
            es.index(index=kas_index_name, body=body)


if __name__ == '__main__':
    index(index_ticket=False, index_kas=True)
