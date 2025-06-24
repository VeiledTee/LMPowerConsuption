from config import CONFIG


def build_prompt(example: dict, include_passage: bool) -> str:
    q = example["question"]

    # Determine dataset type
    dataset_type = "hotpot" if "hotpot" in CONFIG.dataset_name else "boolq"

    # Select appropriate template set
    templates = CONFIG.prompt_templates[dataset_type]

    if include_passage:
        # Handle different context formats per dataset
        if dataset_type == "hotpot":
            context = "\n".join(
                f"{title}: {' '.join(s.strip() for s in sents)}"
                for title, sents in zip(
                    example["context"]["title"], example["context"]["sentences"]
                )
            )
        else:  # boolq
            context = example["context"]

        return templates["with_context"].format(context=context, question=q)
    else:
        return templates["without_context"].format(question=q)