import pandas as pd
import json
from collections import Counter
import sys
import re

# --- Configuration ---
# Input files
ACTIONS_CSV_PATH = 'actions_output2_all.csv'
KEYPHRASES_JSON_PATH = 'external_resources.json' 
ACTION_COLUMN_NAME = 'Extracted Action'

# Output file for the final counts
OUTPUT_CSV_PATH = 'exact_phrase_counts_whole_word.csv'

# Number of words to show before and after the matched phrase in the terminal
CONTEXT_WORD_COUNT = 10 

# --- Terminal Color Codes for Highlighting ---
YELLOW = '\033[93m'
BOLD = '\033[1m'
ENDC = '\033[0m'

def load_keyphrases_from_json(filename: str) -> list[str]:
    """
    Loads the keyword JSON file and extracts all key phrases into a single flat list.
    """
    print(f"Loading key phrases from '{filename}'...")
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        all_phrases = [
            phrase for category_phrases in data.values() for phrase in category_phrases.keys()
        ]
        print(f"Successfully loaded {len(all_phrases)} key phrases.")
        return all_phrases
    except FileNotFoundError:
        print(f"--- ERROR ---: The keyword file was not found at '{filename}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"--- ERROR ---: The file '{filename}' is not a valid JSON file.")
        sys.exit(1)

def enumerate_exact_matches(actions_df: pd.DataFrame, key_phrases: list[str]) -> pd.DataFrame:
    """
    Searches actions for whole-word matches, prints the context of each match
    to the terminal, and returns a DataFrame with total counts.
    """
    print("Enumerating whole-word phrase occurrences...")
    
    phrase_counts = Counter()

    if ACTION_COLUMN_NAME not in actions_df.columns:
        print(f"--- ERROR ---: Column '{ACTION_COLUMN_NAME}' not found in the CSV.")
        sys.exit(1)

    # Iterate with index to potentially reference the source document if needed
    for index, action in actions_df[ACTION_COLUMN_NAME].astype(str).items():
        if not action.strip():
            continue

        for phrase in key_phrases:
            pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
            
            # Use finditer to get match objects, which contain start/end positions
            matches = list(pattern.finditer(action))
            
            if matches:
                # Add the number of matches found in this action to the total count
                phrase_counts[phrase] += len(matches)
                
                # For each match, extract and print the context
                for match in matches:
                    # Get the start and end character positions of the match
                    start, end = match.span()
                    
                    # Extract the text before and after the match
                    before_text = action[:start]
                    after_text = action[end:]
                    
                    # Get the specified number of words for the context
                    before_words = before_text.split()[-CONTEXT_WORD_COUNT:]
                    after_words = after_text.split()[:CONTEXT_WORD_COUNT]
                    
                    # Join the words back into a readable string
                    context_before = ' '.join(before_words)
                    context_after = ' '.join(after_words)
                    
                    # Get the actual text that was matched (to preserve its original case)
                    matched_phrase = match.group(0)

                    # Print the formatted context to the terminal
                    print(
                        f"Found: [{BOLD}{phrase}{ENDC}] in row {index} >> "
                        f"...{context_before} {BOLD}{YELLOW}{matched_phrase}{ENDC} {context_after}..."
                    )

    results_df = pd.DataFrame(phrase_counts.items(), columns=['Key Phrase', 'Count'])
    results_df = results_df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    
    return results_df

def main():
    """
    Main function to run the enumeration process.
    """
    print("--- Starting Task: Whole-Word Phrase Enumeration with Context ---")
    
    key_phrases = load_keyphrases_from_json(KEYPHRASES_JSON_PATH)
    
    try:
        actions_df = pd.read_csv(ACTIONS_CSV_PATH)
        print(f"Successfully loaded {len(actions_df)} actions from '{ACTIONS_CSV_PATH}'.\n")
    except FileNotFoundError:
        print(f"--- ERROR ---: The actions file was not found at '{ACTIONS_CSV_PATH}'")
        return

    final_counts_df = enumerate_exact_matches(actions_df, key_phrases)
    
    final_counts_df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print("\n--- Processing Complete ---")
    print(f"Output saved to '{OUTPUT_CSV_PATH}'")
    print(f"Found {len(final_counts_df[final_counts_df['Count'] > 0])} phrases with at least one whole-word match.")

if __name__ == '__main__':
    main()