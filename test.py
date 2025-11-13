import pandas as pd
from datasets import load_dataset
import json
import os


def extract_nq_long_answers(split_name='validation'):
    """
    Loads the Natural Questions dataset (validation split) and extracts
    the long answer text for examples where a long answer is present.

    Args:
        split_name (str): The dataset split to load (e.g., 'validation').

    Returns:
        pd.DataFrame: A DataFrame containing the question, example ID,
                      and the extracted long answer text.
    """
    print(f"Loading Natural Questions dataset split: '{split_name}'...")
    try:
        # Load the dataset. Set trust_remote_code=True as specified by the dataset's page.
        dataset = load_dataset(
            'google-research-datasets/natural_questions',
            split=split_name,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have the 'datasets' library installed: pip install datasets")
        return pd.DataFrame()

    print(f"Dataset loaded successfully with {len(dataset)} examples.")

    # We will use the 'document_text' and the long_answer indices from the annotations.
    extracted_data = []

    for i, example in enumerate(dataset):

        # --- FIX: Check if annotations exist before accessing index 0 ---
        if not example['annotations']:
            long_answer_text = "N/A - No Annotations Found"
        else:
            # We focus on the first annotation for the ground truth answer in this simple extraction.
            annotation = example['annotations'][0]
            long_answer = annotation['long_answer']

            # Long answer extraction logic:
            # Check if a valid long answer is present (start_token >= 0)
            start_token = long_answer['start_token']
            end_token = long_answer['end_token']

            if start_token >= 0 and end_token > start_token:
                # The 'document_text' field is a list of tokens.
                # We slice the list of tokens based on the start and end indices
                # and then join them back into a single string.
                long_answer_tokens = example['document_text']['tokens'][start_token:end_token]
                long_answer_text = " ".join(long_answer_tokens)
            else:
                # If the first annotation exists but doesn't have a valid long answer span
                long_answer_text = "N/A - No Valid Long Answer Span"

        # Append the results
        extracted_data.append({
            'example_id': example['example_id'],
            'question_text': example['question']['text'],
            'long_answer_text': long_answer_text
        })

        # Display progress every 1000 examples
        if (i + 1) % 1000 == 0:
            print(f"Processing... {i + 1}/{len(dataset)} examples completed.")

    # Convert the list of results into a DataFrame for easy viewing/exporting
    df = pd.DataFrame(extracted_data)
    return df


if __name__ == '__main__':

    results_df = extract_nq_long_answers()

    if not results_df.empty:
        print("\n--- Extracted Long Answers (First 5 examples) ---")
        print(results_df.head().to_markdown(index=False))

