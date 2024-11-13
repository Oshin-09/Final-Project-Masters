from fastapi import FastAPI
from pydantic import BaseModel
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

# Initialize FastAPI app
app = FastAPI()

# Load the LLM model and tokenizer
model_path = 'openlm-research/open_llama_3b_v2'
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map={'': 'cpu'}, low_cpu_mem_usage=True
)

# Define request and response structure
class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
async def root():
    return {"message": "Welcome to the LLM API!"}

@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt = request.prompt
    print("Received prompt:", prompt)  # Debugging print statement
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids=input_ids, max_new_tokens=32, do_sample=True, top_k=50, temperature=0.5)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated response:", response)  # Debugging print statement
    return {"response": response}
