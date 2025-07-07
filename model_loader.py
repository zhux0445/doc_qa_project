
"""Load a local Hugging Face model and wrap it for LangChain."""
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
import torch

# Change to any compatible local model
MODEL_ID = "tiiuae/falcon-7b-instruct"

def load_llm(max_new_tokens: int = 200, temperature: float = 0.7):
    """Return a LangChain-compatible LLM object."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    return HuggingFacePipeline(pipeline=gen_pipe)
