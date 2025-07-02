import torch
from codecarbon import EmissionsTracker
from ollama import generate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from config import CONFIG


def inference_ollama(prompt: str, model_name: str) -> str:
    resp = generate(
        model=model_name,
        prompt=prompt,
        options={
            "temperature": 0.0,
            "top_p": 0.9,
            "stop": ["</s>", "\n\n\n"],
            "num_predict": 10,
        },
        think=False,
    )
    return resp.get("response") or resp["choices"][0]["text"]


def load_model_and_tokenizer(model_name: str) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if CONFIG.device == "cuda" else torch.float32,
        device_map="auto" if CONFIG.device == "cuda" else None,
        trust_remote_code=True,
    ).to(CONFIG.device).eval()
    return tokenizer, model


def track_emissions(project_name: str):
    return EmissionsTracker(project_name=project_name, log_level="error")


def huggingface_inference(prompt, model, tokenizer):
    model_max_ctx = getattr(model.config, "max_position_embeddings", tokenizer.model_max_length)
    max_input_len = max(1, model_max_ctx - CONFIG.max_new_tokens)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
        padding=False,
    ).to(CONFIG.device)

    if inputs.input_ids.shape[1] > model_max_ctx:
        print(f"Truncating input from {inputs.input_ids.shape[1]} to {model_max_ctx}")
        inputs = {k: v[:, -model_max_ctx:] for k, v in inputs.items()}

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        tokens = model.generate(
            **inputs,
            max_new_tokens=CONFIG.max_new_tokens,
            do_sample=False,
            no_repeat_ngram_size=2,
            repetition_penalty=1.5,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    return text


def inference(prompt, model, tokenizer, model_name, run_tag, provider: str):
    project_name = f"{CONFIG.dataset_name.split('/')[-1]}_{model_name}_{run_tag}"

    try:
        with track_emissions(project_name) as tracker:
            if provider == "ollama":
                text = inference_ollama(prompt, model_name)
            elif provider == "huggingface":
                text = huggingface_inference(prompt, model, tokenizer)
            else:
                raise ValueError("Provider must be 'ollama' or 'huggingface'")
    except Exception as e:
        print(f"Error during {provider} inference: {e}")
        raise

    return text, {
        "duration": float(tracker.final_emissions_data.duration),
        "energy_consumed": float(tracker.final_emissions_data.energy_consumed),
        "emissions": float(tracker.final_emissions_data.emissions),
    }

