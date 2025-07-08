from config import CONFIG


def build_prompt(example: dict, include_passage: bool) -> str:
    """
    Construct a prompt for a given QA example using dataset-specific templates.

    Args:
        example (dict): A dictionary containing the question and (optionally) context.
        include_passage (bool): Whether to include the context/passage in the prompt.

    Returns:
        str: The formatted prompt string.
    """
    q = example["question"]

    # Determine dataset type
    dataset_type = "hotpot" if "hotpot" in CONFIG.dataset_name else "boolq"

    # Select appropriate template set
    templates = CONFIG.prompt_templates[dataset_type]

    if include_passage:
        # Handle different context formats per dataset
        if dataset_type == "hotpot":
            # Check if we're using retrieved context
            if "retrieved_context" in example:
                context = example["retrieved_context"]
            else:
                context = "\n".join(
                    f"{title}: {' '.join(s.strip() for s in sents)}"
                    for title, sents in zip(
                        example["context"]["title"], example["context"]["sentences"]
                    )
                )
        else:  # boolq
            context = example.get("retrieved_context", example.get("context", ""))
        print(templates["with_context"].format(context=context, question=q))
        return templates["with_context"].format(context=context, question=q)
    else:
        return templates["without_context"].format(question=q)
