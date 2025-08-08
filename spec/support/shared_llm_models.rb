# Shared context for LLM models that need to be loaded once
RSpec.shared_context "llm_models" do
  # Load GGUF model once and cache at class level
  def gguf_llm
    @@gguf_llm_cache ||= begin
      Candle::LLM.from_pretrained(
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", 
        gguf_file: "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf",
        device: Candle::Device.cpu
      )
    rescue => e
      skip "GGUF model loading failed: #{e.message}"
    end
  end
  
  # Load Phi model once and cache at class level  
  def llm
    @@llm_cache ||= begin
      Candle::LLM.from_pretrained(
        "microsoft/phi-2",
        device: Candle::Device.cpu
      )
    rescue => e
      skip "LLM model loading failed: #{e.message}"
    end
  end
end