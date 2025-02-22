# Introduction

This folder holds the necessary scripts to download the necessary corpora and
create indices for more efficient document retrieval for the FedRAG workflow.

The currently supported corpora are:

1. PubMed
2. Textbooks
3. StatPearls
4. Wikipedia

For index generation we use the [FAISS](https://github.com/facebookresearch/faiss) library.

# How-To-Run

To download the data and prepare the retrieval indices, please run:

```bash
bash ./prepare.sh
```

By default, the bash script will download the `StatePearls` and `Textbooks` corpus and create an index using up to the 
first `100` downloaded chunks. The bash script calls the `prepare.py` script for download and indexing. If you would 
like to **download all four corpora** and **prepare the index using all files** of each corpus, please run the bash script 
as follows:

```bash
bash ./prepare.sh --datasets "pubmed" "statpearls" "textbooks" "wikipedia" --index_num_chunks 0
```

## Corpora

To download the data we use the [MedRAG Toolkit](https://github.com/Teddy-XiongGZ/MedRAG) and the associated MedRAG
dataset repository in [HuggingFace](https://huggingface.co/MedRAG), which holds all the curated corpus.

> \[!NOTE\] For StatPearls, since its content is not allowed to be distributed, we download the data from
> the [NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK430685/)
> and use the chunking script provided by
> the [MedRAG toolkit](https://github.com/Teddy-XiongGZ/MedRAG/blob/main/src/data/statpearls.py)

The `download.py` script is responsible for downloading the corpus and executing the required data chunking operations.

## FAISS Indexer

We use the [FAISS](https://github.com/facebookresearch/faiss) index library for index creation and querying due to its
use simplicity and efficient similarity search.

To map the snippets of each corpus to an embedding space and train and create the corpus index, by default we use the
[all-MiniLM-L6-v2 sentence-transformer model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
which maps sentences & paragraphs to a 384 dimensional dense vector space.

The index is created by the `faiss_indexer.py` script. To build the FAISS indexer, function `build_faiss_index()` takes
as an input the number of chunks (files) upon which the indexer should be built. If the value is set to `0`, then all
files are considered during index creation, else only the first `[:num_chunks]` files are considered. The script
also provides a helper function for document retrieval, _assuming the index has already been generated_.

Specifically, the function `query_faiss_index()` reads the created index for the requested dataset, converts the input
query to its embedded representation and performs a similarity search between the query's embedded vector and the
corpus' snippet vectorized formats.

The current implementation of document retrieval for the FAISS index is built with `IndexIVFFlat` and uses the
`faiss.METRIC_L2` metric, which means that the lower the score of a retrieved document the better, since L2 Distance
measures dissimilarity.

If you want to use another embedding model for generating the embeddings, please change the following properties
in the `faiss_indexer.yaml` file.

```yaml
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embedding_dimension: 384
```
