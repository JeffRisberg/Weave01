import asyncio
from openai import OpenAI
import numpy as np
import weave
from weave import Evaluation
import logging

log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

weave.init('question-answer')


def docs_to_embeddings(docs: list) -> list:
    openai = OpenAI()
    document_embeddings = []
    for doc in docs:
        response = (
            openai.embeddings.create(input=doc, model="text-embedding-3-small")
            .data[0]
            .embedding
        )
        document_embeddings.append(response)
    return document_embeddings


def load_articles():
    # Retrieve the dataset
    dataset_ref = weave.ref('knowledge-articles-data').get()

    return dataset_ref.rows


def load_embeddings(articles):
    docs = [article['contents'] for article in articles]

    return docs_to_embeddings(docs)


def load_queries():
    dataset_ref = weave.ref("hr-support-questions-data").get()

    return dataset_ref.rows


@weave.op()
def get_most_relevant_document(query, articles, article_embeddings):
    openai = OpenAI()
    query_embedding = (
        openai.embeddings.create(input=query, model="text-embedding-3-small")
        .data[0]
        .embedding
    )
    similarities = [
        np.dot(query_embedding, doc_emb)
        / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
        for doc_emb in article_embeddings
    ]
    # Get the index of the most similar document
    most_relevant_doc_index = np.argmax(similarities)
    #import pdb
    #db.set_trace()
    return articles.rows[most_relevant_doc_index]


def infer(queries, articles, article_embeddings):
    # Define any custom scoring function
    @weave.op()
    def match_score1(expected: str, model_output: dict) -> dict:
        # Here is where you'd define the logic to score the model output
        return {'match': expected == model_output['generated_text']}

    @weave.op()
    def function_to_evaluate(query: str):
        print(query)
        most_relevant = get_most_relevant_document(query, articles, article_embeddings)
        print(most_relevant)
        # here's where you would add your LLM call and return the output
        return {'generated_text': most_relevant['contents']}

    # Score your examples using scoring functions
    evaluation = Evaluation(
        dataset=queries, scorers=[match_score1]
    )

    # Start tracking the evaluation
    # Run the evaluation
    asyncio.run(evaluation.evaluate(function_to_evaluate))

    log.info("done")


if __name__ == "__main__":
    articles = load_articles()
    embeddings = load_embeddings(articles)
    queries = load_queries()
    infer(queries, articles, embeddings)

