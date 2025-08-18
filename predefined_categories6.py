import pandas as pd
import json
import nltk
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag

# --- NLTK Data Check ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError as e:
    print("--------------------------------------------------------------------")
    print(f"ERROR: NLTK data not found: {e}")
    print("Please run this command to download the missing resources:")
    print("python3 -m nltk.downloader punkt averaged_perceptron_tagger wordnet stopwords")
    print("--------------------------------------------------------------------")
    exit()

# --- Configuration ---
PROCESSING_MODE = "lemmatization"
INPUT_CSV_PATH = 'All_RAG.csv'
KEYWORD_JSON_PATH = 'keyword_categories_4.json'
OUTPUT_CSV_PATH = 'categorized_actions_output_RAG.csv'
MIN_WORD_COUNT = 3
ACTION_COLUMN_NAME = 'Action'

# --- Custom Stopword Exceptions ---
STOPWORD_EXCEPTIONS = set([  
    'of',
    'on',
    'for',
    'with',
    'against',
    'up',
    'ongoing',
    'regular',
    'provide',
])

# --- Global NLP Objects ---
lemmatizer = WordNetLemmatizer()
base_stop_words = set(stopwords.words('english'))
stop_words = base_stop_words - STOPWORD_EXCEPTIONS

# --- POS tag mapping for WordNet ---
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_text(text, mode='lemmatization'):
    # Tokenize and clean input, then optionally apply POS-aware lemmatization.
    if not isinstance(text, str):
        return []
    # Basic tokenization
    raw_tokens = nltk.word_tokenize(text.lower())
    # Filter alphanumeric and stopwords
    filtered = [tok for tok in raw_tokens if tok.isalnum() and tok not in stop_words]
    if mode == 'lemmatization':
        # POS tagging
        tagged = pos_tag(filtered)
        return [lemmatizer.lemmatize(tok, get_wordnet_pos(tag)) for tok, tag in tagged]
    return filtered


def load_and_preprocess_keywords(filename, mode='lemmatization'):
    # Load JSON category->phrase->weight, preprocess each phrase into tokens. Returns dict: {category: [(tok_tuple, weight), ...]}
    data = json.load(open(filename))
    kw_map = {}
    for category, phrase_dict in data.items():
        processed = []
        for phrase, weight in phrase_dict.items():
            toks = preprocess_text(phrase, mode)
            if toks:
                processed.append((tuple(toks), float(weight)))
        kw_map[category] = processed
    return kw_map


def categorize_action_with_scoring(
    action_text,
    category_keywords,
    mode='lemmatization',
    use_ratio_threshold: bool = False,
    ratio: float = 0.3,
    min_score: float = 0.1
):
    # Categorizes an action by checking for the presence of weighted keyword tokens and applying a relative score threshold.
    
    # Parameters:
    # - action_text: The raw string of the action to be categorized.
    # - category_keywords: A dictionary mapping categories to a list of keyword tuples (token_tuple, weight).
    # - mode: The preprocessing mode ('lemmatization' or 'none') for the action text.
    # - use_ratio_threshold: If True (default), the threshold is calculated as a ratio of the highest score for a given action.
    # - ratio: The ratio of the max score to use as the threshold (default 0.5).
    # - min_score: An absolute minimum score, used as a fallback if ratio thresholding is disabled.

    if not isinstance(action_text, str) or len(action_text.split()) < MIN_WORD_COUNT:
        return []

    # Preprocess the input text to get a set of unique tokens
    tokens = preprocess_text(action_text, mode)
    token_set = set(tokens)

    # Score each category based on keyword presence
    scores = Counter()
    for category, keyword_phrases in category_keywords.items():
        for token_sequence, weight in keyword_phrases:
            # Flexible matching: check if the set of keyword tokens is a subset of the action's tokens. This works for single and multi-word keywords.
            if set(token_sequence).issubset(token_set):
                scores[category] += weight

    if not scores:
        return []

    # Determine the score threshold for categorization
    if use_ratio_threshold:
        max_score = max(scores.values())
        threshold = ratio * max_score
    else:
        threshold = min_score

    # Filter categories that meet or exceed the calculated threshold
    best_categories = [
        category for category, score in scores.items() if score >= threshold
    ]

    return sorted(best_categories)


def main():
    print(f"Starting categorization with mode: '{PROCESSING_MODE}'")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
        return

    if ACTION_COLUMN_NAME not in df.columns:
        print("--- !!! ERROR !!! ---")
        print(f"Column '{ACTION_COLUMN_NAME}' not found. Available columns: {list(df.columns)}")
        return

    # Load and preprocess keywords
    processed_kw = load_and_preprocess_keywords(KEYWORD_JSON_PATH, PROCESSING_MODE)
    # Determine max phrase length for n-grams
    max_ngram = max(
        (len(tok_seq) for kws in processed_kw.values() for tok_seq, _ in kws),
        default=1
    )

    # Categorize actions
    df['category_list'] = df[ACTION_COLUMN_NAME].apply(
        lambda a: categorize_action_with_scoring(a, processed_kw, PROCESSING_MODE, max_ngram)
    )

    # One-hot encode categories
    cat_dums = pd.get_dummies(df['category_list'].explode()).groupby(level=0).sum()
    df = df.join(cat_dums)

    # Flag uncategorized
    if not cat_dums.empty:
        df['Uncategorized'] = (df[cat_dums.columns].sum(axis=1) == 0).astype(int)
    else:
        df['Uncategorized'] = 1

    # Output
    df.drop(columns=['category_list'], inplace=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("\n--- Processing Complete ---")
    print(f"Output saved to '{OUTPUT_CSV_PATH}'")
    print("\nThe output now contains a separate column for each category.")

if __name__ == '__main__':
    main()
