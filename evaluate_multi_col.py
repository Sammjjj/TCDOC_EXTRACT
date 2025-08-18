import pandas as pd
from sklearn.metrics import confusion_matrix

# --- Configuration ---
PREDICTIONS_FILE = 'categorized_actions_output_multicol_weighted2.csv'
GROUND_TRUTH_FILE = 'ground_truth2.csv'
OUTPUT_REPORT_FILE = 'sensitivity_specificity_report_weighted2.csv'

# --- Column Name Configuration ---
PREDICTION_ACTION_COL = 'Extracted Action'
GROUND_TRUTH_ACTION_COL = 'Extracted Action'
GROUND_TRUTH_CATEGORY_COL = 'Ground Truth'

# Normalization function for action text
def normalize_text(s):
    if not isinstance(s, str):
        return ''
    # strip whitespace, collapse multiple spaces, lowercase
    return ' '.join(s.strip().lower().split())


def calculate_metrics(df_merged, category_cols):
    report_data = []

    for category in sorted(category_cols):
        if category not in df_merged.columns:
            print(f"Warning: Predicted category column '{category}' not found. Skipping.")
            continue

        # True labels (0/1) and predicted labels (converted to int)
        y_true = (df_merged[GROUND_TRUTH_CATEGORY_COL] == category).astype(int)
        y_pred = pd.to_numeric(df_merged[category], errors='coerce').fillna(0).astype(int)

        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        except ValueError:
            # handle edge cases with no pos/neg samples
            tn = fp = fn = tp = 0
            if y_true.sum() == 0 and y_pred.sum() == 0:
                tn = len(df_merged)
            elif y_true.sum() == len(df_merged) and y_pred.sum() == len(df_merged):
                tp = len(df_merged)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        report_data.append({
            'Category': category,
            'Sensitivity (Recall)': sensitivity,
            'Specificity': specificity,
            'True Positives': tp,
            'False Positives': fp,
            'True Negatives': tn,
            'False Negatives': fn,
            'Support (Actual Positives)': tp + fn
        })

    return pd.DataFrame(report_data)


def main():
    print("Loading prediction and ground truth files...")
    try:
        df_preds = pd.read_csv(PREDICTIONS_FILE)
        df_gt = pd.read_csv(GROUND_TRUTH_FILE)
    except FileNotFoundError as e:
        print(f"Error: Could not find a file. Check path: {e.filename}")
        return

    # Normalize action text
    df_preds['action_norm'] = df_preds[PREDICTION_ACTION_COL].apply(normalize_text)
    df_gt['action_norm'] = df_gt[GROUND_TRUTH_ACTION_COL].apply(normalize_text)

    # Merge on normalized text
    print("Merging data on normalized columns 'action_norm'...")
    df_merged = pd.merge(
        df_preds,
        df_gt,
        how='inner',
        on='action_norm'
    )

    if df_merged.empty:
        print("Error: Merged dataframe is empty. Check normalization and key columns.")
        return

    print(f"Successfully merged {len(df_merged)} actions after normalization.")

    # Identify predicted category columns
    excluded = set(df_gt.columns) | {PREDICTION_ACTION_COL, 'action_norm'}
    predicted_category_cols = [c for c in df_preds.columns if c not in excluded]

    # Calculate metrics
    report_df = calculate_metrics(df_merged, predicted_category_cols)

    # Output results
    print("\n--- Sensitivity and Specificity Report ---")
    print(report_df.to_string(index=False))
    report_df.to_csv(OUTPUT_REPORT_FILE, index=False)
    print(f"\nReport saved to: {OUTPUT_REPORT_FILE}")

if __name__ == '__main__':
    main()
