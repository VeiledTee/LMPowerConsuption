import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv

load_dotenv()


@dataclass
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
    think: bool
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
    smtp_password: str
    seed: int
    dataset_size: int
    log_level: str = "INFO"
    prompt_templates: dict[str, str] = field(
        default_factory=lambda: {
            "hotpot_qa": {
                "with_context": (
                    "Using the provided context, answer the question with as few words as possible. "
                    "Be thorough in your analysis of the context but answer in as few words as possible. "
                    "Do not overcomplicate your thinking. Do not go in circles.\n"
                    "Context: {context}\n"
                    "Question: {question}\n"
                    "Answer:"
                ),
                "without_context": (
                    "Answer with as few words as possible. "
                    "Be thorough in your analysis but answer in as few words as possible. "
                    "Do not overcomplicate your thinking. Do not go in circles.\n"
                    "Question: {question}\n"
                    "Answer:"
                ),
            },
            "boolq": {
                "with_context": (
                    "Using the provided context, answer the question ONLY with 'True' or 'False'. "
                    "Be thorough in your analysis of the context but answer with just one word. "
                    "Do not overcomplicate your thinking. Do not go in circles.\n"
                    "Context: {context}\n"
                    "Question: {question}\n"
                    "Answer:"
                ),
                "without_context": (
                    "Answer ONLY with 'True' or 'False'. "
                    "Be thorough in your analysis but answer with just one word. "
                    "Do not overcomplicate your thinking. Do not go in circles.\n"
                    "Question: {question}\n"
                    "Answer:"
                ),
            },
            "trivia_qa": {
                "with_context": (
                    "Using the provided context, answer the question with as few words as possible. "
                    "Be thorough in your analysis of the context but answer in as few words as possible. "
                    "Do not overcomplicate your thinking. Do not go in circles.\n"
                    "Context: {context}\n"
                    "Question: {question}\n"
                    "Answer:"
                ),
                "without_context": (
                    "Answer with as few words as possible. "
                    "Be thorough in your analysis but answer in as few words as possible. "
                    "Do not overcomplicate your thinking. Do not go in circles.\n"
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
        "deepseek-r1:1.5b": "ollama",  # doesn't pass boolq baseline
        # "deepseek-r1:7b": "ollama",  # near identical to 8b
        # "deepseek-r1:8b": "ollama",
        # "deepseek-r1:14b": "ollama",
        # "deepseek-r1:32b": "ollama",
        # "gemma3:1b": "ollama",  # doesn't pass boolq baseline
        # "gemma3:4b": "ollama",
        # "gemma3:12b": "ollama",
        # "gemma3:27b": "ollama",
        # "smollm:135m": "ollama",
    },
    modes={
        # "distilbert/distilgpt2": {"q": False, "q+r": True},
        # "openai-community/gpt2-xl": {"q": False},
        # "google/gemma-2b": {"q": False, "q+r": True},
        # "google/gemma-2b-it": {"q": False, "q+r": True},
        # "google/gemma-7b": {"q": False},
        # "google/gemma-7b-it": {"q": False},
        # "meta-llama/Llama-2-7b-hf": {"q": False, "q+r": True},
        # "meta-llama/Llama-2-13b-hf": {"q": False},
        "deepseek-r1:1.5b": {"q+r": True},
        # "deepseek-r1:7b": {"q": False, "q+r": True},
        # "deepseek-r1:8b": {"q": False, "q+r": True},
        # "deepseek-r1:14b": {"q": False, "q+r": True},
        # "deepseek-r1:32b": {"q": False},
        # "gemma3:1b": {"q": False, "q+r": True},
        # "gemma3:4b": {"q+r": True},
        # "gemma3:12b": {"q": False, "q+r": True},
        # "gemma3:27b": {"q": False},
        # "smollm:135m": {"q+r": True},
    },
    # dataset_name="hotpotqa/hotpot_qa",
    dataset_name="mandarjoshi/trivia_qa",
    # dataset_name="google/boolq",
    dataset_file="full.jsonl",  # for full dataset (above) run
    # dataset_file="boolq_mini_dev_128.jsonl",  # for mini boolq dev
    # dataset_file="boolq_mini_128.jsonl",  # for mini boolq test
    # dataset_file="hotpot_mini_dev_128.jsonl",  # for mini hotpot dev
    # dataset_file="hotpot_mini_128.jsonl",  # for mini hotpot test
    # config="fullwiki",
    config="unfiltered",
    # config="unfiltered.nocontext",
    split="validation",
    n_samples=None,
    max_new_tokens=64,
    batch_size=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    think=True,
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
    to_email="s72kw@unb.ca",
    smtp_password=str(os.getenv("SMTP_PASSWORD")),
    seed=42,
    dataset_size=1000,
)
