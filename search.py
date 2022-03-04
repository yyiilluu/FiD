from typing import List, Dict

from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")


def search_es(ticket_subject, ticket_body) -> (List[Dict[str, str]], List[Dict[str, str]]):
    """
    given ticket subject and body,
    return list of ctxs
    """
    ticket_query = {
        "bool": {
            "should": [{"multi_match": {
                "query": ticket_subject + " " + ticket_body,
                "type": "most_fields",
                "fields": ["question_title^2", "question_body"],
                "fuzziness": 4,
                "prefix_length": 0,
                "max_expansions": 300
            }}]
        }
    }
    ticket_results = es.search(index="tickets", body={"query": ticket_query, "from": 0, "size":3})
    print(ticket_results['hits'])

    kas_query = {
        "bool": {
            "should": [{"multi_match": {
                "query": ticket_subject + " " + ticket_body,
                "type": "most_fields",
                "fields": ["document_title^2", "paragraph"],
                "fuzziness": 4,
                "prefix_length": 0,
                "max_expansions": 300
            }}]
        }
    }
    kas_results = es.search(index="knowledge_articles", body={"query": kas_query, "from": 0,
                                                            "size":5})
    return ticket_results, kas_results


def format_ticket_and_kas_into_ctxs(ticket_results, kas_results, ticket_k=1):
    ctxs = []
    ticket_sources = [ t['_source'] for t in ticket_results['hits']['hits'][:ticket_k]]
    for ticket_source in ticket_sources:
        ctxs.append({
            "title": ticket_source['question_title'],
            "text": ticket_source['answer'],
            "id": ticket_source['id'],
            "score": 1,
        })

    for kas_hit in kas_results['hits']['hits']:
        kas_source = kas_hit['_source']
        ctxs.append({
            "title": kas_source['document_title'],
            "text": kas_source['paragraph'].split("\n\n")[1],
            "id": kas_source['id'],
            "score": 1,
        })

    return ctxs

if __name__ == '__main__':
    ticket_results, kas_results = search_es("i need help with taxes",
                                            "i am in california i pay too much tax fix it")

    ctxs = format_ticket_and_kas_into_ctxs(ticket_results, kas_results)
