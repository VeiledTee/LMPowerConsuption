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
    modes: dict[str, bool]
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
        "distilbert/distilgpt2": "huggingface",
        # "openai-community/gpt2-xl": "huggingface",
        # "google/gemma-7b": "huggingface",
        # "google/gemma-7b-it": "huggingface",
        # "google/gemma-2b": "huggingface",
        # "google/gemma-2b-it": "huggingface",
        # "meta-llama/Llama-2-7b-hf": "huggingface",
        # "meta-llama/Llama-2-13b-hf": "huggingface",
    },
    dataset_name="hotpotqa/hotpot_qa",
    # dataset_name="google/boolq",

    dataset_file="boolq_1.jsonl",  # for full dataset (above) run
    # dataset_file="boolq_mini_128.jsonl",  # for mini boolq
    # dataset_file="hotpot_mini_128.jsonl",  # for mini hotpot

    config="fullwiki",
    split="validation",
    n_samples=None,
    max_new_tokens=64,
    batch_size=32,
    # device="cpu",
    device="cuda" if torch.cuda.is_available() else "cpu",
    # modes={"q": False},
    modes={"q+r": True},
    # modes={"q": False, "q+r": True},
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
)
