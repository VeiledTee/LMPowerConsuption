from codecarbon import EmissionsTracker
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import CONFIG


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if CONFIG.device == "cuda" else torch.float32,
            device_map="auto" if CONFIG.device == "cuda" else None,
            trust_remote_code=True,
        )
        .to(CONFIG.device)
        .eval()
    )
    return tokenizer, model


def inference(prompt, model, tokenizer, model_name, run_tag):
    with EmissionsTracker(
        project_name=f"hotpot_{model_name}_{run_tag}", log_level="error"
    ) as tracker:
        with torch.inference_mode():
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer.model_max_length - CONFIG.max_new_tokens,
                padding=False,
            ).to(CONFIG.device)
            tokens = model.generate(
                **inputs,
                max_new_tokens=CONFIG.max_new_tokens,
                do_sample=False,
                no_repeat_ngram_size=2,
                repetition_penalty=1.5,
                eos_token_id=tokenizer.eos_token_id,
            )
    text = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    data = tracker.final_emissions_data
    return text, {
        "duration": float(data.duration),
        "energy_consumed": float(data.energy_consumed),
        "emissions": float(data.emissions),
    }
