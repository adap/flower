"""fedrag: A Flower app."""

from typing import List

from flwr.common.typing import Parameters
from .data.faiss_indexer import Retriever


def str_to_parameters(text: List[str]) -> Parameters:
    tensors = [str.encode(t) for t in text]
    return Parameters(tensors=tensors, tensor_type="string")


def parameters_to_str(parameters: Parameters) -> List[str]:
    text = [param.decode() for param in parameters.tensors]
    return text


def index_exists(corpus_names):
    retriever = Retriever()
    for corpus_name in corpus_names:
        if not retriever.index_exists(corpus_name):
            raise RuntimeError(
                f"Please first download the corpus and create the corresponding index for {corpus_name}."
            )
