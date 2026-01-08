"""
Helper script to prepare baseline and validation datasets for feature range calculation.

This script helps you:
1. Extract genuine reviews from existing training data
2. Create baseline dataset (genuine reviews only)
3. Create validation dataset (genuine + fraudulent reviews)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Directory structure constants
TRAINING_DATA_DIR = "training-data"
BASELINE_DATA_DIR = "baseline-range-data"
VALIDATION_DATA_DIR = "baseline-range-validation"


def get_base_models_dir():
    """Get the base models directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def get_training_data_dir():
    """Get the training data directory"""
    base_dir = get_base_models_dir()
    data_dir = os.path.join(base_dir, TRAINING_DATA_DIR)
    return data_dir


def get_baseline_data_dir():
    """Get the baseline data directory"""
    base_dir = get_base_models_dir()
    data_dir = os.path.join(base_dir, BASELINE_DATA_DIR)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


def get_validation_data_dir():
    """Get the validation data directory"""
    base_dir = get_base_models_dir()
    data_dir = os.path.join(base_dir, VALIDATION_DATA_DIR)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


def prepare_baseline_dataset(source_file=None, min_reviews=500):
    """
    Prepare baseline dataset of genuine reviews.
    
    Args:
        source_file: Path to source CSV file. If None, uses processed_reviews.csv
        min_reviews: Minimum number of reviews required
    """
    print("=" * 80)
    print("PREPARING BASELINE DATASET (Genuine Reviews Only)")
    print("=" * 80)
    
    training_dir = get_training_data_dir()
    baseline_dir = get_baseline_data_dir()
    
    # Load source data
    if source_file is None:
        source_file = os.path.join(training_dir, 'processed_reviews.csv')
    
    if not os.path.exists(source_file):
        print(f"\nError: Source file not found: {source_file}")
        print("\nPlease provide a CSV file with 'review' or 'original_review' column")
        return
    
    print(f"\nLoading data from: {source_file}")
    df = pd.read_csv(source_file)
    print(f"✓ Loaded {len(df)} reviews")
    
    # Determine review column
    if 'original_review' in df.columns:
        review_col = 'original_review'
    elif 'review' in df.columns:
        review_col = 'review'
    else:
        print("\nError: No 'review' or 'original_review' column found")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Filter for genuine reviews
    # Assuming 'label' column exists with 'OR' or 'Normal' for genuine reviews
    if 'label' in df.columns:
        # Map common label values
        genuine_labels = ['OR', 'Normal', 'Genuine', '0', 0]
        genuine_df = df[df['label'].isin(genuine_labels)].copy()
        print(f"✓ Found {len(genuine_df)} genuine reviews (from label column)")
    else:
        # If no label column, use all reviews (assume all are genuine)
        print("  Warning: No 'label' column found. Using all reviews as genuine.")
        genuine_df = df.copy()
    
    # Check minimum size
    if len(genuine_df) < min_reviews:
        print(f"\nWarning: Only {len(genuine_df)} genuine reviews found.")
        print(f"  Minimum recommended: {min_reviews} reviews")
        print(f"  Consider adding more genuine reviews for better statistical validity")
        
        response = input(f"\nContinue with {len(genuine_df)} reviews? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Prepare baseline dataset
    baseline_df = pd.DataFrame({
        'review': genuine_df[review_col].values
    })
    
    # Add label if available
    if 'label' in genuine_df.columns:
        baseline_df['label'] = 'Genuine'
    
    # Save baseline dataset
    output_file = os.path.join(baseline_dir, 'baseline_genuine_reviews.csv')
    baseline_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Baseline dataset saved to: {output_file}")
    print(f"  Total reviews: {len(baseline_df)}")
    print(f"  Columns: {baseline_df.columns.tolist()}")
    
    return baseline_df


def prepare_validation_dataset(source_file=None, genuine_count=700, fraudulent_count=300):
    """
    Prepare validation dataset with both genuine and fraudulent reviews.
    
    Args:
        source_file: Path to source CSV file. If None, uses processed_reviews.csv
        genuine_count: Number of genuine reviews to include
        fraudulent_count: Number of fraudulent reviews to include
    """
    print("\n" + "=" * 80)
    print("PREPARING VALIDATION DATASET (Genuine + Fraudulent Reviews)")
    print("=" * 80)
    
    training_dir = get_training_data_dir()
    validation_dir = get_validation_data_dir()
    
    # Load source data
    if source_file is None:
        source_file = os.path.join(training_dir, 'processed_reviews.csv')
    
    if not os.path.exists(source_file):
        print(f"\nError: Source file not found: {source_file}")
        return
    
    print(f"\nLoading data from: {source_file}")
    df = pd.read_csv(source_file)
    print(f"✓ Loaded {len(df)} reviews")
    
    # Determine review column
    if 'original_review' in df.columns:
        review_col = 'original_review'
    elif 'review' in df.columns:
        review_col = 'review'
    else:
        print("\nError: No 'review' or 'original_review' column found")
        return
    
    # Separate genuine and fraudulent reviews
    if 'label' in df.columns:
        # Map labels
        genuine_labels = ['OR', 'Normal', 'Genuine', '0', 0]
        fraudulent_labels = ['CG', 'Anomalous', 'Fraudulent', 'Fake', '1', 1]
        
        genuine_df = df[df['label'].isin(genuine_labels)].copy()
        fraudulent_df = df[df['label'].isin(fraudulent_labels)].copy()
        
        print(f"✓ Found {len(genuine_df)} genuine reviews")
        print(f"✓ Found {len(fraudulent_df)} fraudulent reviews")
    else:
        print("\nError: No 'label' column found. Cannot separate genuine/fraudulent reviews.")
        print("  Please ensure your dataset has a 'label' column with values:")
        print("    - Genuine: 'OR', 'Normal', 'Genuine', '0'")
        print("    - Fraudulent: 'CG', 'Anomalous', 'Fraudulent', 'Fake', '1'")
        return
    
    # Sample reviews
    if len(genuine_df) < genuine_count:
        print(f"\nWarning: Only {len(genuine_df)} genuine reviews available.")
        print(f"  Requested: {genuine_count}, Using: {len(genuine_df)}")
        genuine_sample = genuine_df
    else:
        genuine_sample = genuine_df.sample(n=genuine_count, random_state=42)
    
    if len(fraudulent_df) < fraudulent_count:
        print(f"\nWarning: Only {len(fraudulent_df)} fraudulent reviews available.")
        print(f"  Requested: {fraudulent_count}, Using: {len(fraudulent_df)}")
        fraudulent_sample = fraudulent_df
    else:
        fraudulent_sample = fraudulent_df.sample(n=fraudulent_count, random_state=42)
    
    # Combine into validation dataset
    validation_df = pd.DataFrame({
        'review': pd.concat([
            genuine_sample[review_col],
            fraudulent_sample[review_col]
        ]).values,
        'label': pd.concat([
            pd.Series(['Genuine'] * len(genuine_sample)),
            pd.Series(['Fraudulent'] * len(fraudulent_sample))
        ]).values
    })
    
    # Shuffle
    validation_df = validation_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save validation dataset
    output_file = os.path.join(validation_dir, 'validation_dataset.csv')
    validation_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Validation dataset saved to: {output_file}")
    print(f"  Total reviews: {len(validation_df)}")
    print(f"  Genuine: {len(genuine_sample)}")
    print(f"  Fraudulent: {len(fraudulent_sample)}")
    print(f"  Columns: {validation_df.columns.tolist()}")
    
    return validation_df


def main():
    """Main function to prepare both datasets."""
    print("\n" + "=" * 80)
    print("DATASET PREPARATION FOR FEATURE RANGE CALCULATION")
    print("=" * 80)
    
    print("\nThis script will help you prepare:")
    print("  1. Baseline dataset: Genuine reviews only (for range calculation)")
    print("  2. Validation dataset: Genuine + Fraudulent reviews (for validation)")
    
    # Prepare baseline dataset
    baseline_df = prepare_baseline_dataset()
    
    if baseline_df is not None:
        # Prepare validation dataset
        validation_df = prepare_validation_dataset()
        
        print("\n" + "=" * 80)
        print("DATASET PREPARATION COMPLETE")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Review the generated datasets")
        print("  2. Run: python feature_range_calculation.py")
        print("  3. Review the generated range report")
    else:
        print("\nBaseline dataset preparation failed. Please check your data.")


if __name__ == "__main__":
    main()

