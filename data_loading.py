"""
data_loading.py — loads the raw financial-QA JSONL dataset and preprocesses
it into a unified (question, text, table, answer, metadata) record shape.

Expected input format: JSONL, one record per line, each with fields like
`question`, `pre_text`, `context`, `post_text`, `table`, `original_answer`,
`company_name`, `report_year`, `company_sector`.
"""

import json
import re
from io import StringIO

import pandas as pd
from tqdm import tqdm


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def clean_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def parse_table(table_str):
    """Parses a raw CSV-like table string into a markdown table (better for LLM context)."""
    try:
        df = pd.read_csv(StringIO(table_str))
        return df.to_markdown(index=False)
    except Exception:
        return str(table_str)


def prepare_dataset(data, max_samples=500):
    processed = []
    for item in tqdm(data[:max_samples], desc="Preprocessing"):
        try:
            full_text = " ".join(filter(None, [
                clean_text(item.get("pre_text", "")),
                clean_text(item.get("context", "")),
                clean_text(item.get("post_text", "")),
            ])).strip()

            processed.append({
                "question": clean_text(item.get("question", "")),
                "text": full_text,
                "table": parse_table(item.get("table", "")),
                "answer": clean_text(item.get("original_answer", "")),
                "metadata": {
                    "company": item.get("company_name"),
                    "year": item.get("report_year"),
                    "sector": item.get("company_sector"),
                },
            })
        except Exception as e:
            print(f"Skipping sample: {e}")
    return processed
