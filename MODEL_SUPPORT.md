# Model Support in red-candle

## Known Working Models

| Model | Model Path | GGUF File | Tokenizer | Initializer |
| :----- | :---- | :---- | :---- | :---- |
| Mistral | `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` | `mistral-7b-instruct-v0.2.Q4_K_M.gguf` | | `llm = Candle::LLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", gguf_file: "mistral-7b-instruct-v0.2.Q4_K_M.gguf")` |
| TinyLlama | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` | `tinyllama-1.1b-chat-v1.0.Q4_0.gguf` | | `llm = Candle::LLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", gguf_file: "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")` |
| Gemma 3 | `google/gemma-3-4b-it-qat-q4_0-gguf` | `gemma-3-4b-it-q4_0.gguf` | `google/gemma-3-4b-it` | `llm = Candle::LLM.from_pretrained("google/gemma-3-4b-it-qat-q4_0-gguf", gguf_file: "gemma-3-4b-it-q4_0.gguf", tokenizer: "google/gemma-3-4b-it")` |
| Phi-2 | `microsoft/phi-2` | `phi-2.Q4_K_M.gguf` | | `llm = Candle::LLM.from_pretrained("microsoft/phi-2")` |
| Phi-2 | `TheBloke/phi-2-GGUF` | `phi-2.Q4_K_M.gguf` | | `llm = Candle::LLM.from_pretrained("TheBloke/phi-2-GGUF", gguf_file: "phi-2.Q4_K_M.gguf")` |
| Phi-3 | `microsoft/Phi-3-mini-4k-instruct` | `Phi-3-mini-4k-instruct-q4.gguf` | | `llm = Candle::LLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")` |


## ⚠️ Partially Working Models

| Model | Model Path | GGUF File | Tokenizer | Initializer | Status |
| :----- | :---- | :---- | :---- | :---- | :---- |
| Phi-3 | `microsoft/Phi-3-mini-4k-instruct-gguf` | `Phi-3-mini-4k-instruct-q4.gguf` | | `llm = Candle::LLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct-gguf", gguf_file: "Phi-3-mini-4k-instruct-q4.gguf")` | works once, fails on subsequent calls, [candle PR](https://github.com/huggingface/candle/pull/2937) |
| Phi-4 | `microsoft/phi-4-gguf` | `phi-4-Q4_K_S.gguf` | | `llm = Candle::LLM.from_pretrained("microsoft/phi-4-gguf", gguf_file: "phi-4-Q4_K_S.gguf")` | works once, fails on subsequent calls, [candle PR](https://github.com/huggingface/candle/pull/2937) |

