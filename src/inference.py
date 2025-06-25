import torch
from codecarbon import EmissionsTracker
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from config import CONFIG
from typing import Tuple, Dict


def load_model_and_tokenizer(model_name: str) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Load a Hugging Face tokenizer and causal language model for inference.

    Args:
        model_name (str): The name or path of the model to load.

    Returns:
        Tuple[PreTrainedTokenizer, PreTrainedModel]: Loaded tokenizer and model.
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
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    run_tag: str,
) -> Tuple[str, Dict[str, float]]:
    """
    Run inference with emissions tracking and return generated text with energy metrics.

    Args:
        prompt (str): Input prompt for the model.
        model (PreTrainedModel): Loaded model for inference.
        tokenizer (PreTrainedTokenizer): Tokenizer associated with the model.
        model_name (str): Name of the model (for logging purposes).
        run_tag (str): Tag identifying the run (used in emissions log naming).

    Returns:
        Tuple[str, Dict[str, float]]: Generated text and energy/emissions data.
    """
    try:
        with EmissionsTracker(
            project_name=f"{CONFIG.dataset_name.split('/')[-1]}_{model_name}_{run_tag}",
            log_level="error",
        ) as tracker:
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
                    print(f"Truncating from {inputs.input_ids.shape[1]} to {model_max_ctx}")
                    inputs = {k: v[:, -model_max_ctx:] for k, v in inputs.items()}

                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id

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
        return text, {
            "duration": float(tracker.final_emissions_data.duration),
            "energy_consumed": float(tracker.final_emissions_data.energy_consumed),
            "emissions": float(tracker.final_emissions_data.emissions),
        }
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise
