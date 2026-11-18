# Retrieval-Augmented Generation (RAG)

Retrieval-augmented generation (RAG) is a technique where a language model answers a question
using passages retrieved from an external knowledge source at query time, instead of relying
only on what it memorised during training. The term was introduced by Lewis et al. (2020) at
Facebook AI Research.

## How a RAG pipeline works

A RAG system has two phases.

**Indexing (offline).** Documents are loaded, split into chunks (typically a few hundred to a
thousand characters, with some overlap so sentences are not cut off from their context), and
each chunk is converted into a vector by an embedding model. The vectors are stored in a
vector database such as Chroma, Qdrant, pgvector or FAISS, together with metadata like the
source file name.

**Retrieval and generation (online).** The user's question is embedded with the same
embedding model. The vector store returns the top-k chunks whose vectors are most similar
(usually by cosine similarity). These chunks are inserted into the prompt as context, and the
language model generates an answer grounded in them, ideally citing which chunk each claim
came from.

## Why RAG reduces hallucinations

Hallucination is when a model produces fluent but false or unsupported statements. RAG reduces
it because the model is given relevant, up-to-date evidence and is instructed to answer only
from that evidence. It also makes answers verifiable: citations let a reader check the source.
RAG does not eliminate hallucination. If retrieval returns irrelevant chunks, or the model
ignores the context, it can still invent facts. This is why RAG systems are evaluated for
*faithfulness* (is the answer supported by the retrieved context?) and *relevance*.

## RAG versus fine-tuning

Fine-tuning changes a model's weights and is good for teaching style, format or a narrow skill.
RAG keeps knowledge outside the model, so updating knowledge only requires re-indexing
documents, with no retraining. RAG is generally cheaper for frequently changing knowledge and
gives traceable sources; the two approaches can be combined.

## Common improvements

- **Hybrid search**: combine keyword search (BM25) with vector search, which helps with exact
  terms such as product codes or names.
- **Reranking**: retrieve more candidates (e.g. top 20) and reorder them with a cross-encoder
  reranker before keeping the best few.
- **Query rewriting**: have the model rewrite or decompose the question before retrieval.
- **Agentic RAG**: let an agent decide when and what to retrieve, possibly over several steps,
  by exposing retrieval as a tool.
