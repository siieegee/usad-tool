import json
import os
import pandas as pd


FEATURE_COLUMNS = [
    'review_length', 'lexical_diversity', 'avg_word_length',
    'sentiment_polarity', 'sentiment_subjectivity', 'word_entropy',
    'repetition_ratio', 'exclamation_count', 'question_count',
    'capital_ratio', 'punctuation_density'
]


def compute_basis(df: pd.DataFrame) -> dict:
    basis = {}
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if series.empty:
            continue
        p10 = float(series.quantile(0.10))
        p90 = float(series.quantile(0.90))
        mean = float(series.mean())
        basis[col] = {
            'min': round(p10, 6),
            'max': round(p90, 6),
            'optimal': round(mean, 6)
        }
    return basis


def main():
    # Expect train_data.csv produced by feature_extraction.py in the working dir
    input_path = os.path.join(os.path.dirname(__file__) or '.', 'train_data.csv')
    if not os.path.exists(input_path):
        raise FileNotFoundError('train_data.csv not found. Run feature_extraction.py first.')

    df = pd.read_csv(input_path)
    basis = compute_basis(df)

    output_path = os.path.join(os.path.dirname(__file__) or '.', 'feature_basis.json')
    with open(output_path, 'w') as f:
        json.dump({'normal': basis}, f, indent=2)

    print(f"Saved percentile-based feature basis to: {output_path}")


if __name__ == '__main__':
    main()


