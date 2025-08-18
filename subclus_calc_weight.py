import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# --- 1. SETUP ---
# Load the dataset
try:
    df = pd.read_csv('Actions_FinalData - All_Data.csv')
except FileNotFoundError:
    print("Error: 'Actions_FinalData - All_Data.csv' not found. Make sure the file is in the correct directory.")
    exit()

# Get a list of all unique categories
unique_categories = df['Categories'].unique()

# Open a file to write the results to
output_filename = 'initial_keyword_analysis.txt'
with open(output_filename, 'w') as f:
    print(f"Starting analysis... Results will be saved to '{output_filename}'")
    f.write("--- Top 15 Keywords for Each Category ---\n")
    f.write("This file can be used to identify keywords for weighted clustering.\n")
    f.write("="*60 + "\n\n")

    # --- 2. LOOP THROUGH CATEGORIES & EXTRACT KEYWORDS ---
    # Iterate over each unique category in the dataset
    for category in unique_categories:
        # Filter the dataframe for the current category
        category_df = df[df['Categories'] == category]
        statements = category_df['Extracted Action'].tolist()

        # Ensure there's at least one statement to analyze
        if len(statements) < 1:
            continue

        # Initialize the TF-IDF Vectorizer - smooth_idf=True and sublinear_tf=True for better scoring
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            smooth_idf=True,
            sublinear_tf=True
        )

        try:
            # Generate the TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(statements)
            feature_names = vectorizer.get_feature_names_out()

            # Sum the TF-IDF scores for each term across all documents in the category
            summed_tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()

            # Get the indices of the top 15 terms
            top_indices = summed_tfidf_scores.argsort()[-15:][::-1]

            # Get the names of the top 15 terms
            top_keywords = [feature_names[i] for i in top_indices]

            # --- 3. WRITE TO FILE ---
            # Write the category and its top keywords to the file
            f.write(f"## Category: {category}\n")
            f.write(', '.join(top_keywords) + '\n\n')

        except ValueError:
            # This handles cases where a category's documents contain only stop words
            f.write(f"## Category: {category}\n")
            f.write("Could not identify keywords (likely only contains stop words).\n\n")


print("Analysis complete.")