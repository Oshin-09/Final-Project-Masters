from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import uvicorn
from fuzzywuzzy import process, fuzz
import os
import torch.autograd.profiler as profiler
import time

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GPT-Neo-API")

# Check device
device = "mps" if torch.backends.mps.is_built() else "cpu"
logger.info(f"Using device: {device}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-1.3B",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
).to(device)

PRE_CACHED_ANSWERS = {
    "What is a CT scan?": "A CT scan is an imaging technique that uses X-rays and computer processing to create detailed cross-sectional images of the body.",
    "What does a CT scan detect?": "A CT scan detects abnormalities such as tumors, infections, fractures, internal injuries, and other medical conditions.",
    "How long does a CT scan take?": "A CT scan usually takes 10 to 30 minutes, depending on the area being scanned.",
    "Is a CT scan painful?": "No, a CT scan is a painless procedure.",
    "Is contrast dye used in a CT scan?": "Contrast dye may be used to enhance the visibility of certain structures or abnormalities.",
    "What are pulmonary nodules?": "Pulmonary nodules are small growths in the lungs that can be benign or malignant.",
    "What does 'follow-up recommended' mean?": "It means additional scans or evaluations are advised to monitor changes over time.",
    "Can a CT scan confirm cancer?": "A CT scan can identify suspicious areas but cannot confirm cancer; a biopsy is often needed.",
    "What are 'findings' in a CT scan report?": "Findings are observations made by the radiologist based on the images, describing any abnormalities or normal conditions.",
    "What does 'no significant abnormalities' mean?": "It means no serious or noteworthy issues were detected during the scan.",
    "What is a contrast-enhanced CT scan?": "It is a CT scan performed with contrast dye to improve visualization of blood vessels and specific tissues.",
    "Are there any risks with CT scans?": "CT scans use ionizing radiation, which may slightly increase the risk of cancer, especially with repeated exposure. Contrast dye may cause allergic reactions in rare cases.",
    "Why would a doctor recommend a CT scan?": "A doctor may recommend a CT scan to diagnose injuries, infections, or diseases, monitor treatment progress, or guide procedures like biopsies.",
    "Can pregnant women undergo a CT scan?": "CT scans are generally avoided during pregnancy due to potential risks to the fetus, unless absolutely necessary.",
    "What is the difference between a CT scan and an MRI?": "A CT scan uses X-rays, while MRI uses magnetic fields and radio waves to create detailed images. MRI is often preferred for soft tissue evaluation.",
    "What is the significance of Hounsfield Units (HU) in a CT scan?": "Hounsfield Units (HU) measure the density of tissues in a CT scan. For example, water is 0 HU, bone is around 1000 HU, and air is approximately -1000 HU.",
    "What is the purpose of a CT scan of the chest?": "A chest CT scan is used to detect lung diseases, infections, tumors, blood clots, or other abnormalities in the chest area.",
    "What is the difference between a standard CT scan and a low-dose CT scan?": "A low-dose CT scan uses less radiation than a standard CT scan and is commonly used for lung cancer screening.",
    "What does a CT angiography (CTA) show?": "CTA provides detailed images of blood vessels, helping detect blockages, aneurysms, or other vascular abnormalities.",
    "How do I prepare for a CT scan?": "You may need to fast for a few hours, avoid certain medications, and inform your doctor about any allergies or pregnancy. Instructions vary based on the type of scan.",
    "What does 'contrast extravasation' mean?": "It refers to the leakage of contrast dye into surrounding tissues, which can cause swelling and discomfort but is usually not serious.",
    "What does 'calcification' mean in a CT scan report?": "Calcification refers to the accumulation of calcium salts in tissues, often indicating old injuries, inflammation, or certain diseases.",
    "What is the role of CT scans in detecting fractures?": "CT scans provide detailed images of bones, making them useful for diagnosing complex fractures that may not be visible on X-rays.",
    "Can a CT scan detect blood clots?": "Yes, a CT scan, especially a CT pulmonary angiography, can detect blood clots in the lungs or other blood vessels.",
    "What is a CT scan with 3D reconstruction?": "This is a CT scan where the images are processed to create 3D models, aiding in surgical planning or detailed analysis of structures.",
    "What is the difference between a CT scan and an X-ray?": "An X-ray provides 2D images, while a CT scan combines multiple X-rays to create detailed 3D images of the body.",
    "Can a CT scan detect infections?": "Yes, a CT scan can help detect abscesses, pneumonia, or other infections by showing inflammation, fluid accumulation, or other signs.",
    "What does 'mass effect' mean in a CT scan report?": "Mass effect refers to the displacement of tissues or structures due to a mass, such as a tumor or large lesion.",
    "What are the common side effects of contrast dye?": "Common side effects include a warm sensation, metallic taste, or mild allergic reactions. Severe reactions are rare."
}


# Request model
class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the GPT-Neo Medical Question API!"}

@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        question = request.question.strip().lower()
        logger.info(f"Checking for pre-cached response for question: '{question}'")

        # Use fuzzy matching
        best_match, score = process.extractOne(question, PRE_CACHED_ANSWERS.keys(), scorer=fuzz.ratio)
        if score > 80:  # Threshold for similaritysource llama_env/bin/activate

            logger.info(f"Fuzzy match found: '{best_match}' with score {score}")
            return {"answer": PRE_CACHED_ANSWERS[best_match]}

        # Prepare prompt and inputs
        role_instruction = (
            "You are a cautious, well-informed medical assistant. "
            "Only answer based on known medical facts and imaging knowledge. "
        )

        prompt = f"""{role_instruction}

        Question: {request.question}
        Answer:"""

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        # Log inputs for debugging
        logger.info(f"Tokenized input IDs: {inputs['input_ids']}")
        logger.info(f"Attention mask: {inputs['attention_mask']}")

        # Generate response
        start_time = time.time()
        max_new_tokens = 100 
        #max_new_tokens = 50 #for speed
        with profiler.profile(use_cuda=False) as prof:
            response = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample= True,
                temperature=0.7,  # Lower temperature for more deterministic output
                top_k=50,
                num_beams=3, # use 1 for speed
                repetition_penalty=2.0,
            )
        logger.info(prof.key_averages().table(sort_by="cpu_time_total"))
        logger.info(f"Response Time: {time.time() - start_time} seconds")


        # Log raw tokens
        logger.info(f"Raw generated tokens: {response}")

        # Decode response
        decoded_tokens = response[0].clone().detach().cpu().tolist()

        def remove_redundancy(text):
            sentences = text.split(". ")  # Split the text into sentences
            seen = set()
            filtered = []
            for sentence in sentences:
                clean_sentence = sentence.strip().lower()  # Normalize sentence
                if clean_sentence not in seen:
                    filtered.append(sentence.strip())
                    seen.add(clean_sentence)
            return ". ".join(filtered).strip()
        

        try:
            answer = tokenizer.decode(decoded_tokens, skip_special_tokens=True).strip()
            
        except Exception as decode_error:
            logger.error(f"Error decoding tokens: {decoded_tokens}, Error: {decode_error}")
            answer = "The model generated invalid tokens. Please try again."

        answer = remove_redundancy(answer)

        if not answer:
            logger.warning(f"Blank response for question: {request.question}")
            answer = (
                "I'm sorry, I couldn't generate a meaningful answer. "
                "Please try rephrasing your question."
            ) 
            

        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating the answer: {str(e)}")

if __name__ == "__main__":
    # Disable malloc stack logging and tokenizer parallelism warnings
    os.environ["MallocStackLogging"] = "0"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    uvicorn.run(app, host="127.0.0.1", port=8000)