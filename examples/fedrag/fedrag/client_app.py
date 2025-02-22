"""fedrag: A Flower app."""

import os

from flwr.client import ClientApp
from flwr.common import ConfigsRecord, Context, Message, RecordSet

from .data.faiss_indexer import Retriever

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Flower ClientApp
app = ClientApp()


@app.query()
def query(msg: Message, context: Context):

    node_id = context.node_id

    # Extract question
    question = str(msg.content.configs_records["config"]["question"])
    question_id = str(msg.content.configs_records["config"]["question_id"])

    # Extract corpus name
    corpus_name = str(msg.content.configs_records["config"]["corpus_name"])

    print(
        "ClientApp: {} - Question ID: {} - Using Corpus: {}.".format(
            node_id, question_id, corpus_name
        )
    )

    # Initialize retrieval system
    retriever = Retriever()
    # Use the knn value for retrieving the closest-k documents to the query
    knn = int(msg.content.configs_records["config"]["knn"])
    retrieved_docs = retriever.query_faiss_index(corpus_name, question, knn)

    # Create lists with the computed scores and documents
    scores = [doc["score"] for doc_id, doc in retrieved_docs.items()]
    documents = [doc["content"] for doc_id, doc in retrieved_docs.items()]
    print("ClientApp: {} - Retrieved: {} documents.".format(node_id, len(documents)))

    # Create reply record with retrieved documents.
    docs_n_scores = ConfigsRecord(
        {
            "documents": documents,
            "scores": scores,
        }
    )
    reply_record = RecordSet(configs_records={"docs_n_scores": docs_n_scores})

    # Return message
    return msg.create_reply(reply_record)
