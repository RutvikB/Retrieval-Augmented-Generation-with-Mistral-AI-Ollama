# Retrieval Augmented Generation (RAG) using Mistral-7B LLM on Ollama

Implementation of a RAG Application by using MistralAI's <b>Mistral-7B v0.3</b> Pre-trained Large Language Model (with 7.25 Billion Parameters) released on May 22, 2024 hosted on [Ollama](https://ollama.com/library/mistral).

Context Retrieval is achieved by passing relevant documents - <b>Web URLs</b> - in the appropriate context file.

### Environment Variables needed to run code:
- [Mistral AI API Key](https://docs.mistral.ai/api/) 
- [Hugging Face User Access Token](https://huggingface.co/docs/hub/en/security-tokens)
- [LangSmith API Key](https://docs.smith.langchain.com/how_to_guides/setup/create_account_api_key)

## Test Run
The RAG application test run was provided with a context document URL from USA Today dated Aug 12, 2024 which suggests the discovery of Water on Planet Mars ([link](https://www.usatoday.com/story/news/nation/2024/08/12/liquid-water-discovered-on-mars-study/74765921007/))

![Test RAG Output](https://github.com/RutvikB/Retrieval-Augmented-Generation-with-Mistral-AI-Ollama/blob/main/web-rag.gif)
