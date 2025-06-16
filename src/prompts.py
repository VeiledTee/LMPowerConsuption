from typing import Dict, Any
from config import CONFIG


def build_prompt(example: Dict[str, Any], include_passage: bool) -> str:
    q = example["question"]
    if include_passage:
        context = "\n".join(
            f"{title}: {' '.join(s.strip() for s in sents)}"
            for title, sents in zip(
                example["context"]["title"], example["context"]["sentences"]
            )
        )
        return CONFIG.prompt_templates["with_context"].format(
            context=context, question=q
        )
    else:
        return CONFIG.prompt_templates["without_context"].format(question=q)
