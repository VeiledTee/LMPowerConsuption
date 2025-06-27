from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass(frozen=True)
class ExperimentConfig:
    model_types: dict[str, str]
    dataset_name: str
    dataset_file: str
    config: str
    split: str
    n_samples: Optional[int]
    max_new_tokens: int
    batch_size: int
    device: str
    modes: dict[str, dict]
    wiki_dir: Path
    corpus_cache: Path
    tfidf_cache: Path
    index_cache: Path
    intro_min_chars: int
    hash_bits: int
    token_pattern: str
    energy_dir: Path
    result_dir: Path
    data_dir: Path
    retrieval_only: bool
    email_results: bool
    from_email: str
    to_email: str
    log_level: str = "INFO"
    prompt_templates: dict[str, str] = field(
        default_factory=lambda: {
            "hotpot": {
                "with_context": (
                    "Answer the following to the best of your ability. You must provide an answer. "
                    "If you are unsure, make an educated guess based on what you know and the context provided. "
                    "Context: {context}\nQuestion: {question}\nAnswer:"
                ),
                "without_context": (
                    "Answer the following to the best of your ability. You must provide an answer. "
                    "If you are unsure, make an educated guess based on what you know. "
                    "Question: {question}\nAnswer:"
                ),
            },
            "boolq": {
                "with_context": (
                    "Read the following passage carefully and answer the question with only one word. It must be 'True' or 'False'.\n\n"
                    "Passage: {context}\n"
                    "Question: {question}\n"
                    "Answer:"
                ),
                "without_context": (
                    "Answer the following question with only one word. It must be 'True' or 'False'.\n"
                    "Question: {question}\n"
                    "Answer:"
                ),
            },
        }
    )


CONFIG = ExperimentConfig(
    model_types={
        # "distilbert/distilgpt2": "huggingface",
        # "openai-community/gpt2-xl": "huggingface",
        # "google/gemma-7b": "huggingface",
        # "google/gemma-7b-it": "huggingface",
        # "google/gemma-2b": "huggingface",
        # "google/gemma-2b-it": "huggingface",
        # "meta-llama/Llama-2-7b-hf": "huggingface",
        # "meta-llama/Llama-2-13b-hf": "huggingface",
        #"deepseek-r1:1.5b": "ollama",
        "deepseek-r1:8b": "ollama",
        "deepseek-r1:14b": "ollama",
        "deepseek-r1:32b": "ollama",
        #"smollm:135m": "ollama",
    },
    # dataset_name="hotpotqa/hotpot_qa",
    dataset_name="google/boolq",
    # dataset_file="boolq_1.jsonl",  # for full dataset (above) run
    dataset_file="boolq_mini_128.jsonl",  # for mini boolq
    # dataset_file="hotpot_mini_128.jsonl",  # for mini hotpot
    config="fullwiki",
    split="validation",
    n_samples=None,
    max_new_tokens=64,
    batch_size=8,
    device="cuda" if torch.cuda.is_available() else "cpu",
    modes={
        # "distilbert/distilgpt2": {"q": False, "q+r": True},
        # "openai-community/gpt2-xl": {"q": False},
        # "google/gemma-2b": {"q": False, "q+r": True},
        # "google/gemma-2b-it": {"q": False, "q+r": True},
        # "google/gemma-7b": {"q": False},
        # "google/gemma-7b-it": {"q": False},
        # "meta-llama/Llama-2-7b-hf": {"q": False, "q+r": True},
        # "meta-llama/Llama-2-13b-hf": {"q": False},
        #"deepseek-r1:1.5b": {"q": False, "q+r": True},
        "deepseek-r1:8b": {"q": False, "q+r": True},
        "deepseek-r1:14b": {"q": False, "q+r": True},
        "deepseek-r1:32b": {"q": False},
        #"smollm:135m": {"q+r": True},
    },
    wiki_dir=Path("data/hotpot_wiki-processed"),
    corpus_cache=Path("cache/wiki.pkl"),
    tfidf_cache=Path("cache/tfidf.pkl"),
    index_cache=Path("cache/index.pkl"),
    intro_min_chars=51,
    hash_bits=20,
    token_pattern=r"(?u)\b\w+\b",
    energy_dir=Path(__file__).resolve().parent.parent / "results" / "energy",
    result_dir=Path(__file__).resolve().parent.parent / "results",
    data_dir=Path(__file__).resolve().parent.parent / "data",
    retrieval_only=False,
    email_results=True,
    from_email="eheavey626@gmail.com",
    to_email="s72kw@unb.com",
)
