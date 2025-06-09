from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

app = FastAPI()

# Chemin ou repo HF de ton adaptateur (exemple: repo HF public ou local)
adapter_path = "BK20145/llama-usmba-fes-refined-final"  # Remplace par ton repo HF

# Charger la config PEFT
peft_config = PeftConfig.from_pretrained(adapter_path)

# Charger le modèle de base
base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, torch_dtype=torch.float16, device_map="auto")

# Charger le modèle avec les poids LoRA (adaptateur)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Classe pour la requête
class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100

@app.post("/generate/")
async def generate_text(request: GenerationRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=request.max_new_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": text}

