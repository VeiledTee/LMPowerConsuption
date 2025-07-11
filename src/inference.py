import logging
import os

import torch
from codecarbon import EmissionsTracker
from ollama import generate
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizer)

from config import CONFIG

logger = logging.getLogger("codecarbon")
logger.setLevel(logging.ERROR)
logger.propagate = False

# Remove existing handlers
for handler in logger.handlers:
    logger.removeHandler(handler)


def inference_ollama(prompt, model_name):
    resp = generate(
        model=model_name,
        prompt=prompt,
        options={
            "temperature": 0.0,
            "top_p": 0.9,
            "stop": ["</s>", "\n\n\n"],
            "num_thread": os.cpu_count()
        },
        think=CONFIG.think,
    )

    if "response" in resp:
        return resp["response"]
    if "choices" in resp and len(resp["choices"]) > 0:
        return resp["choices"][0]["text"]
    raise ValueError(f"Invalid Ollama response for prompt: {prompt[:200]}...\nResponse: {resp}")


def load_model_and_tokenizer(
    model_name: str,
) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Load a Hugging Face tokenizer and causal language model for inference.

    Args:
        model_name (str): The name or path of the model to load.

    Returns:
        tuple[PreTrainedTokenizer, PreTrainedModel]: Loaded tokenizer and model.
    """
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


def inference(
    prompt: str,
    model_name: str,
    run_tag: str,
    provider: str,
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
):
    """
    Run inference with emissions tracking and return generated text with energy metrics.

    Args:
        prompt (str): Input prompt for the model.
        model (PreTrainedModel): Loaded model for inference.
        tokenizer (PreTrainedTokenizer): Tokenizer associated with the model.
        model_name (str): Name of the model (for logging purposes).
        run_tag (str): Tag identifying the run (used in emissions log naming).
        provider (str): The service providing the language model.

    Returns:
        tuple[str, dict[str, float]]: Generated text and energy/emissions data.
    """
    if provider == "ollama":
        try:
            with EmissionsTracker(
                save_to_file=False,
                project_name=f"{CONFIG.dataset_name.split('/')[-1]}_{model_name}_{run_tag}",
                log_level="error",
            ) as tracker:
                text = inference_ollama(prompt, model_name)

            return text, {
                "duration": float(tracker.final_emissions_data.duration),
                "energy_consumed": float(tracker.final_emissions_data.energy_consumed),
                "emissions": float(tracker.final_emissions_data.emissions),
            }
        except Exception as e:
            print(f"Error during {provider} inference: {str(e)}")
            raise
    elif provider == "huggingface":
        try:
            with torch.inference_mode():
                model_max_ctx = getattr(
                    model.config, "max_position_embeddings", tokenizer.model_max_length
                )
                max_length_val = max(1, model_max_ctx - CONFIG.max_new_tokens)

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length_val,
                    padding=False,
                ).to(CONFIG.device)

                if inputs.input_ids.shape[1] > model_max_ctx:
                    print(
                        f"Truncating from {inputs.input_ids.shape[1]} to {model_max_ctx}"
                    )
                    inputs = {k: v[:, -model_max_ctx:] for k, v in inputs.items()}

                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
            with EmissionsTracker(
                    save_to_file=False,
                    project_name=f"{CONFIG.dataset_name.split('/')[-1]}_{model_name}_{run_tag}",
                    log_level="error",
            ) as tracker:
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
            print(f"Inference: {tracker.final_emissions_data.duration}s")
            return text, {
                "duration": float(tracker.final_emissions_data.duration),
                "energy_consumed": float(tracker.final_emissions_data.energy_consumed),
                "emissions": float(tracker.final_emissions_data.emissions),
            }
        except Exception as e:
            print(f"Error during {provider} inference: {str(e)}")
            raise
    else:
        print(
            f"Error during inference: Provider in CONFIG.model_types must be either 'ollama' or 'huggingface'."
        )
        raise EnvironmentError
