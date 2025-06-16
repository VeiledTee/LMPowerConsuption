from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch


@dataclass(frozen=True)
class ExperimentConfig:
    model_candidates: List[str]
    dataset_name: str
    config: str
    split: str
    n_samples: Optional[int]
    max_new_tokens: int
    batch_size: int
    device: str
    modes: Dict[str, bool]
    wiki_dir: Path
    corpus_cache: Path
    tfidf_cache: Path
    index_cache: Path
    intro_min_chars: int
    hash_bits: int
    token_pattern: str
    energy_dir: Path
    result_dir: Path
    retrieval_only: bool
    log_level: str = "INFO"
    prompt_templates: Dict[str, str] = field(
        default_factory=lambda: {
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
        }
    )


CONFIG = ExperimentConfig(
    model_candidates=[
        "google/gemma-7b-it",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
    ],
    dataset_name="hotpotqa/hotpot_qa",
    config="fullwiki",
    split="validation",
    n_samples=None,
    max_new_tokens=64,
    batch_size=32,
    device="cpu",  # "cuda" if torch.cuda.is_available() else "cpu"
    modes={"q": False},
    wiki_dir=Path("data/enwiki-processed"),
    corpus_cache=Path("cache/wiki.pkl"),
    tfidf_cache=Path("cache/tfidf.pkl"),
    index_cache=Path("cache/index.pkl"),
    intro_min_chars=51,
    hash_bits=20,
    token_pattern=r"(?u)\b\w+\b",
    energy_dir=Path(__file__).resolve().parent.parent / "results" / "energy",
    result_dir=Path(__file__).resolve().parent.parent / "results",
    retrieval_only=False,
)
