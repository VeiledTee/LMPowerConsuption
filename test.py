from datasets import load_dataset
from bs4 import BeautifulSoup


def extract_first_paragraph(html_content):
    """Extract the first paragraph from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    first_p = soup.find('p')
    return first_p.get_text().strip() if first_p else ""


# Load the dataset
dataset = load_dataset('google-research-datasets/natural_questions', split='train')

# Process examples and extract first paragraphs
for example in dataset:
    first_paragraph = extract_first_paragraph(example['document']['html'])

    print(f"ID: {example['id']}")
    print(f"Question: {example['question']['text']}")
    print(f"First Paragraph: {first_paragraph}")
    print(f"URL: {example['document']['url']}")
    print("-" * 80)