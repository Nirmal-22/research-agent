# Evaluating LLM Applications

LLM applications are non-deterministic and produce open-ended text, so traditional unit tests
are not enough. Teams combine several evaluation methods.

## Kinds of evaluation

**Deterministic checks.** Cheap, exact tests on the output: is it valid JSON, does it contain
required fields, are citation numbers valid, is it under a length limit, does it avoid banned
content. These should run on every change, like unit tests.

**Reference-based metrics.** Compare the output to a known correct answer. Exact match works for
classification; for free text, metrics like ROUGE or embedding similarity are used, though they
correlate poorly with human judgement for long answers.

**LLM-as-a-judge.** A language model grades the output against a rubric, for example scoring
from 1 to 5 how faithful an answer is to its sources. It scales far better than human review.
Known weaknesses: position bias, verbosity bias (preferring longer answers), self-preference
(favouring outputs from the same model family) and inconsistency between runs. Mitigations
include clear rubrics, asking for a short justification before the score, using a stronger
judge model, and calibrating the judge against a sample of human labels.

**Human evaluation.** The gold standard for quality, but slow and expensive. Usually used to
build a labelled dataset and to validate automated judges.

## RAG-specific metrics

- **Faithfulness (groundedness)**: is every claim in the answer supported by the retrieved
  context? Low faithfulness means hallucination.
- **Answer relevance**: does the answer address the question that was asked?
- **Context precision and recall**: did retrieval return the relevant chunks, and only those?

Frameworks such as Ragas, DeepEval, TruLens, LangSmith and Langfuse implement these metrics.

## Evaluation workflow

1. Build a small dataset of realistic inputs (20 to 100 examples is a good start), with
   reference answers or key facts where possible.
2. Run the application on the dataset and record outputs, latency and token usage.
3. Score outputs with deterministic checks and an LLM judge.
4. Track scores across versions to catch regressions when prompts, models or retrieval
   settings change. Running evals in CI turns them into regression tests.
