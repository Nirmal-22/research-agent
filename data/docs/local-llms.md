# Running LLMs Locally with Ollama

Ollama is an open-source tool for downloading and running language models on a local machine.
It wraps the llama.cpp inference engine, manages model files, and exposes an HTTP API on
port 11434. Models are pulled by name and tag, for example `ollama pull qwen3:4b-instruct`.

## Quantization and memory

Local models are usually distributed in the GGUF format and *quantized*: weights are stored
with fewer bits (for example 4-bit Q4_K_M instead of 16-bit floats). Quantization cuts memory
use roughly by a factor of three to four with a modest quality loss. A rough rule: a 4-bit
model needs about 0.6 to 0.7 GB of memory per billion parameters, plus memory for the
context window (the KV cache), which grows with the context length. A 4-billion-parameter
model at 4-bit therefore fits comfortably on an 8 GB GPU.

## Small models and their limits

Models with 3 to 8 billion parameters are fast and private, but compared to large hosted
models they:

- follow complex instructions less reliably,
- make more mistakes in tool calling (wrong arguments, calling tools in a loop, or answering
  without calling a tool at all),
- produce invalid JSON more often unless constrained decoding is used,
- have less world knowledge, so they benefit strongly from retrieval (RAG).

Practical mitigations: keep prompts short and explicit, give tools simple signatures, use
*structured outputs* (Ollama accepts a JSON schema in the `format` field and constrains
generation to match it), cap the number of agent iterations, retry on parse errors, and
always have a fallback path.

## Thinking models

Some models, such as Qwen3 and DeepSeek-R1, can produce a hidden reasoning trace before the
answer ("thinking"). Thinking can improve accuracy on hard problems but increases latency and
token usage. Ollama exposes a `think` option to turn it off for models that support both
modes; models released as thinking-only variants always think.

## Alternatives

Other local serving options include llama.cpp directly, LM Studio, and vLLM. vLLM is designed
for high-throughput GPU serving with continuous batching and PagedAttention, and is common in
production deployments of open-weight models.
