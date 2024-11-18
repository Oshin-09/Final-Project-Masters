from fastapi import FastAPI
from pydantic import BaseModel
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
from fastapi.concurrency import run_in_threadpool
import logging

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the LLM model and tokenizer
model_path = 'openlm-research/open_llama_3b_v2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map={'': 'cpu'} if device == 'cpu' else 'auto', 
    low_cpu_mem_usage=True
)

# Pre-warm the model
@app.lifespan("startup")
async def warm_up_model():
    logging.info("Warming up the model...")
    _ = model.generate(tokenizer("Warm-up", return_tensors="pt").input_ids)

# Define request and response structure
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 32

@app.get("/")
async def root():
    return {"message": "Welcome to the LLM API!"}

@app.post("/generate")
async def generate_text(request: PromptRequest):
    if not request.prompt.strip():
        return {"error": "Prompt cannot be empty."}
    if len(request.prompt) > 512:
        return {"error": "Prompt is too long. Please limit it to 512 characters."}

    logging.info(f"Received prompt: {request.prompt}")
    input_ids = tokenizer(request.prompt, return_tensors="pt").input_ids
    output = await run_in_threadpool(
        model.generate, input_ids=input_ids, max_new_tokens=request.max_tokens, do_sample=True, top_k=50, temperature=0.5
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    logging.info(f"Generated response: {response}")
    return {"response": response}
