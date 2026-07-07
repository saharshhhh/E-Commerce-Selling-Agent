import os
import pandas as pd
from typing import List, Dict

def load_orders(csv_path: str) -> pd.DataFrame:
    """Load orders from CSV."""
    return pd.read_csv(csv_path)

def load_docs(docs_dir: str) -> List[Dict[str, str]]:
    """Load and chunk Markdown policy documents."""
    chunks = []
    if not os.path.exists(docs_dir):
        return []
    for filename in os.listdir(docs_dir):
        if filename.endswith(".md"):
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple chunking by double newline (paragraphs/sections)
                sections = content.split('\n\n')
                current_header = ""
                for section in sections:
                    if section.startswith('#'):
                        current_header = section.split('\n')[0]

                    chunks.append({
                        "source": filename,
                        "header": current_header,
                        "text": section.strip()
                    })
    return chunks
