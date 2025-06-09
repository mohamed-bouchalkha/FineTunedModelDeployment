from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# URL ou repo Hugging Face de ton modèle LoRA (adapter + base model)
adapter_repo = "BK20145/llama-usmba-fes-refined-final"

# Charger la config PEFT depuis Hugging Face Hub
peft_config = PeftConfig.from_pretrained(adapter_repo)

# Charger le modèle de base (par exemple LLaMA 2) directement depuis Hugging Face Hub
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.float16,  # ou "auto"
    device_map="auto"           # pour utiliser le GPU si dispo
)

# Charger les poids LoRA depuis ton repo (adapter)
model = PeftModel.from_pretrained(base_model, adapter_repo)

# Charger le tokenizer depuis ton repo aussi
tokenizer = AutoTokenizer.from_pretrained(adapter_repo)

# Générer une réponse
prompt = "Parle-moi de l'université Sidi Mohamed Ben Abdellah."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  # envoyer sur même device que le modèle

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
