"""
rag_alignment_workflow_unpaired.py

Modified workflow for UNPAIRED action lists from extraction output.

Key features:
- Expects CSV with columns: LineID, Institution, Document Type, Action, RAG value
- Task 1: Categorise all actions using 26-category taxonomy
- Task 2: Within-period matching (AP1 vs RAG_AP1) using fast TF-IDF
- Task 3: Cross-period matching (RAG_AP1 → AP2) using Gemini AI for intelligent analysis
- Outputs each action with its matches
- **NEW: Checkpoint saving and resume functionality**

Author: Sam's Research Project
Date: December 2025
"""

import json
import os
import re
import time
from collections import deque

import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Vertex AI for Gemini (optional but recommended for cross-period matching)
try:
    import vertexai
    from vertexai.language_models import TextEmbeddingModel

    VERTEXAI_IMPORTED = True
except ImportError:
    VERTEXAI_IMPORTED = False
    print(
        "WARNING: vertexai library not installed. Cross-period matching will use TF-IDF fallback."
    )

# ==================== CONFIGURATION ====================

# Google Cloud / Vertex AI settings
PROJECT_ID = "tcdocprog"
LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
GEMINI_MODEL = "gemini-2.0-flash-001"

# Rate limiting
USE_RATE_LIMITING = True  # ENABLED to avoid hitting API quotas
REQUESTS_PER_MINUTE = 15  # Conservative limit for free tier
BATCH_COMPARISON_SIZE = 10  # Reduced from 15 to avoid hitting token limits
BATCH_PAUSE_SECONDS = 4  # Increased pause between batches

# File paths
INPUT_CSV_PATH = "rag_actions_extraction.csv"
OUTPUT_CSV_PATH = "rag_actions_extraction_MATCHED.csv"
KEYWORD_JSON_PATH = "keyword_categories_5.json"

# Checkpoint paths
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_STATE_FILE = os.path.join(CHECKPOINT_DIR, "workflow_state.json")

# Task 2: Within-period matching thresholds (simple, fast TF-IDF)
WITHIN_PERIOD_IDENTICAL_THRESHOLD = 0.95  # >= this = "Identical"
WITHIN_PERIOD_MATCH_THRESHOLD = 0.85  # >= this = "Match"
WITHIN_PERIOD_SIMILAR_THRESHOLD = 0.70  # >= this = "Similar"

# Task 3: Cross-period matching parameters (intelligent Gemini)
TOP_N_MATCHES = 5  # Number of top matches to report
FUZZY_MIN_SCORE = 0.5  # Minimum score to include in top 5

# Categorisation settings
PROCESSING_MODE = "lemmatization"
MIN_WORD_COUNT = 3

# Custom stopword exceptions
STOPWORD_EXCEPTIONS = set(
    ["of", "on", "for", "with", "against", "up", "ongoing", "regular", "provide"]
)

# Global caches
EMBEDDING_CACHE = {}
GEMINI_RAW_RESPONSES = []

# Rate limiting tracking
REQUEST_TIMESTAMPS = deque(maxlen=REQUESTS_PER_MINUTE)


# ==================== CHECKPOINT MANAGEMENT ====================


def ensure_checkpoint_dir():
    """Create checkpoint directory if it doesn't exist."""
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"✓ Created checkpoint directory: {CHECKPOINT_DIR}")


def save_checkpoint(df, state_info, checkpoint_name="checkpoint"):
    """
    Save current dataframe and workflow state.

    Args:
        df: Current dataframe
        state_info: Dictionary with state information (task, institution, periods, etc.)
        checkpoint_name: Name for this checkpoint
    """
    ensure_checkpoint_dir()

    # Save dataframe
    csv_checkpoint = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.csv")
    df.to_csv(csv_checkpoint, index=False)

    # Save state
    state_info["checkpoint_name"] = checkpoint_name
    state_info["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(CHECKPOINT_STATE_FILE, "w") as f:
        json.dump(state_info, f, indent=2)

    print(f"  ✓ Checkpoint saved: {checkpoint_name}")


def load_checkpoint():
    """
    Load the most recent checkpoint.

    Returns:
        tuple: (df, state_info) or (None, None) if no checkpoint exists
    """
    if not os.path.exists(CHECKPOINT_STATE_FILE):
        return None, None

    try:
        # Load state
        with open(CHECKPOINT_STATE_FILE, "r") as f:
            state_info = json.load(f)

        checkpoint_name = state_info.get("checkpoint_name", "checkpoint")
        csv_checkpoint = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.csv")

        if not os.path.exists(csv_checkpoint):
            print(f"  Warning: Checkpoint CSV not found: {csv_checkpoint}")
            return None, None

        # Load dataframe
        df = pd.read_csv(csv_checkpoint)

        print(f"✓ Loaded checkpoint: {checkpoint_name}")
        print(f"  Saved at: {state_info.get('timestamp', 'unknown time')}")
        print(f"  Last completed: {state_info.get('last_completed', 'unknown')}")

        return df, state_info

    except Exception as e:
        print(f"  Error loading checkpoint: {e}")
        return None, None


def clear_checkpoints():
    """Remove all checkpoint files."""
    if os.path.exists(CHECKPOINT_DIR):
        for file in os.listdir(CHECKPOINT_DIR):
            file_path = os.path.join(CHECKPOINT_DIR, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"  Warning: Could not remove {file_path}: {e}")
        print("✓ Cleared all checkpoints")


def rate_limit_sleep():
    """Sleep if necessary to stay within rate limits."""
    if not USE_RATE_LIMITING:
        return

    current_time = time.time()

    while REQUEST_TIMESTAMPS and current_time - REQUEST_TIMESTAMPS[0] > 60:
        REQUEST_TIMESTAMPS.popleft()

    if len(REQUEST_TIMESTAMPS) >= REQUESTS_PER_MINUTE:
        sleep_time = 60 - (current_time - REQUEST_TIMESTAMPS[0]) + 1
        if sleep_time > 0:
            print(
                f"    [Rate limit] Sleeping {sleep_time:.1f}s to stay within quota..."
            )
            time.sleep(sleep_time)

    REQUEST_TIMESTAMPS.append(time.time())


# ==================== NLTK SETUP ====================

NLTK_AVAILABLE = False
lemmatizer = None
stop_words = None

try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("taggers/averaged_perceptron_tagger")
    nltk.data.find("corpora/wordnet")
    nltk.data.find("corpora/stopwords")

    lemmatizer = WordNetLemmatizer()
    base_stop_words = set(stopwords.words("english"))
    stop_words = base_stop_words - STOPWORD_EXCEPTIONS
    NLTK_AVAILABLE = True
    print("✓ NLTK data loaded successfully")
except LookupError as e:
    print("=" * 70)
    print(f"WARNING: NLTK data not found: {e}")
    print("Categorisation will be skipped. Matching will proceed normally.")
    print("To enable categorisation, run:")
    print(
        "  python3 -m nltk.downloader punkt averaged_perceptron_tagger wordnet stopwords"
    )
    print("=" * 70)
    NLTK_AVAILABLE = False

# ==================== VERTEX AI SETUP ====================

EMBEDDINGS_AVAILABLE = False
GEMINI_AVAILABLE = False
embedding_model = None
gemini_model = None

if VERTEXAI_IMPORTED:
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        # Try to initialise embedding model
        try:
            embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
            EMBEDDINGS_AVAILABLE = True
            print(f"✓ Embeddings initialised: {EMBEDDING_MODEL}")
        except Exception as e:
            print(f"  Warning: Could not initialise embeddings: {e}")

        # Try to initialise Gemini generative model
        try:
            from vertexai.generative_models import GenerativeModel

            gemini_model = GenerativeModel(GEMINI_MODEL)
            GEMINI_AVAILABLE = True
            print(f"✓ Gemini model initialised: {GEMINI_MODEL}")
        except Exception as e:
            print(f"  Warning: Could not initialise Gemini: {e}")

    except Exception as e:
        print("=" * 70)
        print(f"WARNING: Could not initialise Vertex AI: {e}")
        print("Cross-period matching will fall back to TF-IDF")
        print("=" * 70)
else:
    print("=" * 70)
    print("WARNING: Vertex AI library not available")
    print("Cross-period matching will use TF-IDF")
    print("=" * 70)


def get_embedding(text):
    """Get Gemini embedding for a text string with caching."""
    if not EMBEDDINGS_AVAILABLE or embedding_model is None:
        return None

    if not isinstance(text, str) or not text.strip():
        return None

    cache_key = f"EMBED:{text}"
    if cache_key in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[cache_key]

    try:
        embeddings = embedding_model.get_embeddings([text])
        embedding_vector = np.array(embeddings[0].values)
        EMBEDDING_CACHE[cache_key] = embedding_vector
        return embedding_vector
    except Exception as e:
        print(f"  Warning: Failed to get embedding: {e}")
        return None


def compare_actions_with_gemini_batch(
    rag_action,
    ap_actions_dict,
    rag_categories=None,
    ap_categories_dict=None,
    use_cache=True,
):
    """
    Use Gemini to compare one RAG action against multiple AP actions in a single API call.
    This is MUCH more efficient than individual comparisons.
    """
    if not GEMINI_AVAILABLE or gemini_model is None:
        return {}

    if not isinstance(rag_action, str) or not rag_action.strip():
        return {}

    if not ap_actions_dict:
        return {}

    # Build numbered list for Gemini
    lineid_map = {}  # Maps numbers (1, 2, 3...) to LineIDs
    ap_list_text = ""

    for i, (lineid, ap_text) in enumerate(ap_actions_dict.items(), 1):
        lineid_map[i] = lineid

        # Get categories for this AP action
        ap_cats = ap_categories_dict.get(lineid, []) if ap_categories_dict else []
        cat_str = f" [Categories: {'; '.join(ap_cats)}]" if ap_cats else ""

        ap_list_text += f"\n{i}. {ap_text}{cat_str}"

    # Format RAG categories
    rag_cat_str = ""
    if rag_categories and isinstance(rag_categories, (list, tuple)):
        rag_cat_str = f"\n[Categories: {'; '.join(rag_categories)}]"

    prompt = f"""Compare this RAG assessment action against the following action plan actions:

RAG ACTION:{rag_cat_str}
{rag_action}

ACTION PLAN ACTIONS:{ap_list_text}

For each action plan action, determine the relationship to the RAG action:
1. "Identical" - Essentially the same action
2. "Extended" - RAG action evolved into more ambitious version
3. "Narrowed" - RAG action became more focused
4. "Related" - Connected but different focus
5. "Unrelated" - No meaningful connection

Return JSON with this exact format:
{{
  "matches": [
    {{"number": 1, "relationship": "Extended", "score": 0.85}},
    {{"number": 3, "relationship": "Related", "score": 0.65}}
  ]
}}

Only include matches with score >= 0.5. Order by score (highest first). Include up to 5 matches."""

    try:
        rate_limit_sleep()

        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()

        GEMINI_RAW_RESPONSES.append(
            {"prompt": prompt, "response": response_text, "timestamp": time.time()}
        )

        # Parse JSON
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            return {}

        parsed = json.loads(json_match.group(0))
        matches_list = parsed.get("matches", [])

        # Convert back to LineID-based dictionary
        result = {}
        for match in matches_list:
            number = match.get("number")
            if number and number in lineid_map:
                lineid = lineid_map[number]
                relationship = match.get("relationship", "Related")
                score = match.get("score", 0.5)
                result[lineid] = {"relationship": relationship, "score": score}

        return result

    except Exception as e:
        print(f"    Warning: Gemini comparison failed: {e}")
        return {}


def get_wordnet_pos(tag):
    """Convert NLTK POS tag to WordNet POS tag."""
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_text(text, mode="lemmatization"):
    """Preprocess text with stopword removal and lemmatization/stemming."""
    if not isinstance(text, str):
        return ""

    if not NLTK_AVAILABLE or lemmatizer is None or stop_words is None:
        return text.lower()

    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stop_words]

    if mode == "lemmatization":
        pos_tags = pos_tag(tokens)
        tokens = [
            lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags
        ]

    return " ".join(tokens)


def load_and_preprocess_keywords(json_path, mode="lemmatization"):
    """Load and preprocess keywords from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    processed = {}
    for category, keywords in data.items():
        processed[category] = [preprocess_text(kw, mode) for kw in keywords]

    return processed


def categorise_action_with_scoring(
    action_text, category_keywords, mode="lemmatization"
):
    """Categorise an action using keyword matching with scoring."""
    if not isinstance(action_text, str) or not action_text.strip():
        return []

    processed_action = preprocess_text(action_text, mode)
    action_words = set(processed_action.split())

    if len(action_words) < MIN_WORD_COUNT:
        return []

    scores = {}
    for category, keywords in category_keywords.items():
        score = 0
        for keyword in keywords:
            keyword_words = set(keyword.split())
            if keyword_words.issubset(action_words):
                score += len(keyword_words)

        if score > 0:
            scores[category] = score

    if not scores:
        return []

    max_score = max(scores.values())
    top_categories = [cat for cat, score in scores.items() if score == max_score]

    return sorted(top_categories)


def parse_document_type(doc_type_str):
    """Parse document type string into (type, period)."""
    if not isinstance(doc_type_str, str):
        return (None, None)

    doc_type_str = doc_type_str.strip()

    # Try RAG_AP pattern
    m = re.match(r"RAG_AP(\d+)", doc_type_str, re.IGNORECASE)
    if m:
        return ("RAG_AP", int(m.group(1)))

    # Try AP pattern
    m = re.match(r"AP(\d+)", doc_type_str, re.IGNORECASE)
    if m:
        return ("AP", int(m.group(1)))

    return (None, None)


def match_within_period_simple(df, institution, period):
    """
    Match AP vs RAG_AP within a single period using simple TF-IDF.
    These should be mostly identical or very similar.
    """
    print(f"\n  {institution} - Period {period}:")

    inst_mask = df["Institution"] == institution
    ap_mask = inst_mask & (df["Document Type"] == f"AP{period}")
    rag_mask = inst_mask & (df["Document Type"] == f"RAG_AP{period}")

    ap_df = df[ap_mask]
    rag_df = df[rag_mask]

    if len(ap_df) == 0 or len(rag_df) == 0:
        print("    No data for both AP and RAG_AP")
        return df

    # Build lookup
    ap_lookup = {}
    for idx in ap_df.index:
        lineid = df.at[idx, "LineID"]
        action = df.at[idx, "Action"]
        if pd.notna(lineid) and pd.notna(action):
            ap_lookup[lineid] = action

    rag_lookup = {}
    for idx in rag_df.index:
        lineid = df.at[idx, "LineID"]
        action = df.at[idx, "Action"]
        if pd.notna(lineid) and pd.notna(action):
            rag_lookup[lineid] = action

    if not ap_lookup or not rag_lookup:
        print("    No valid actions to match")
        return df

    # TF-IDF
    all_texts = list(ap_lookup.values()) + list(rag_lookup.values())
    all_ids = list(ap_lookup.keys()) + list(rag_lookup.keys())

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    ap_size = len(ap_lookup)
    rag_size = len(rag_lookup)

    ap_vectors = tfidf_matrix[:ap_size]
    rag_vectors = tfidf_matrix[ap_size:]

    similarity_matrix = cosine_similarity(ap_vectors, rag_vectors)

    # Match AP → RAG_AP
    ap_matches = 0
    for i, ap_lineid in enumerate(list(ap_lookup.keys())):
        similarities = similarity_matrix[i]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score >= WITHIN_PERIOD_IDENTICAL_THRESHOLD:
            status = "Identical"
        elif best_score >= WITHIN_PERIOD_MATCH_THRESHOLD:
            status = "Match"
        elif best_score >= WITHIN_PERIOD_SIMILAR_THRESHOLD:
            status = "Similar"
        else:
            continue

        rag_lineid = list(rag_lookup.keys())[best_idx]
        match_str = f"{rag_lineid} ({status}, {best_score:.2f})"

        ap_idx = df[df["LineID"] == ap_lineid].index[0]
        df.at[ap_idx, "Match AP vs RAG_AP (LineID, Status)"] = match_str
        ap_matches += 1

    # Match RAG_AP → AP
    rag_matches = 0
    for j, rag_lineid in enumerate(list(rag_lookup.keys())):
        similarities = similarity_matrix[:, j]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score >= WITHIN_PERIOD_IDENTICAL_THRESHOLD:
            status = "Identical"
        elif best_score >= WITHIN_PERIOD_MATCH_THRESHOLD:
            status = "Match"
        elif best_score >= WITHIN_PERIOD_SIMILAR_THRESHOLD:
            status = "Similar"
        else:
            continue

        ap_lineid = list(ap_lookup.keys())[best_idx]
        match_str = f"{ap_lineid} ({status}, {best_score:.2f})"

        rag_idx = df[df["LineID"] == rag_lineid].index[0]
        df.at[rag_idx, "Match RAG_AP vs AP (LineID, Status)"] = match_str
        rag_matches += 1

    print(f"    AP→RAG_AP: {ap_matches} matches")
    print(f"    RAG_AP→AP: {rag_matches} matches")

    return df


def match_cross_period_intelligent(df, institution, period_from, period_to):
    """
    Match RAG_AP(x) → AP(x+1) using Gemini AI for intelligent relationship analysis.
    """
    print(f"\n  Matching {institution}: RAG_AP{period_from} → AP{period_to}")

    inst_mask = df["Institution"] == institution
    rag_mask = inst_mask & (df["Document Type"] == f"RAG_AP{period_from}")
    ap_mask = inst_mask & (df["Document Type"] == f"AP{period_to}")

    rag_df = df[rag_mask]
    ap_df = df[ap_mask]

    if len(rag_df) == 0 or len(ap_df) == 0:
        print("    No data for comparison")
        return df

    # Build lookups
    rag_lookup = {}
    rag_categories = {}
    for idx in rag_df.index:
        lineid = df.at[idx, "LineID"]
        action = df.at[idx, "Action"]
        category = df.at[idx, "Category"] if "Category" in df.columns else None

        if pd.notna(lineid) and pd.notna(action):
            rag_lookup[lineid] = action
            if pd.notna(category):
                rag_categories[lineid] = [c.strip() for c in str(category).split(";")]

    ap_lookup = {}
    ap_categories = {}
    for idx in ap_df.index:
        lineid = df.at[idx, "LineID"]
        action = df.at[idx, "Action"]
        category = df.at[idx, "Category"] if "Category" in df.columns else None

        if pd.notna(lineid) and pd.notna(action):
            ap_lookup[lineid] = action
            if pd.notna(category):
                ap_categories[lineid] = [c.strip() for c in str(category).split(";")]

    if not rag_lookup or not ap_lookup:
        print("    No valid actions to match")
        return df

    print(
        f"    Processing {len(rag_lookup)} RAG actions against {len(ap_lookup)} AP actions"
    )

    col_name = f"Fuzzy Match RAG_AP{period_from} → AP{period_to} (LineID, Relationship)"

    # Use Gemini batch comparison if available
    if GEMINI_AVAILABLE:
        total_rag = len(rag_lookup)
        processed = 0
        start_time = time.time()

        for rag_lineid, rag_action in rag_lookup.items():
            processed += 1

            rag_cats = rag_categories.get(rag_lineid, [])

            # Get matches for this RAG action
            matches = compare_actions_with_gemini_batch(
                rag_action, ap_lookup, rag_cats, ap_categories
            )

            if matches:
                # Format top 5 matches
                sorted_matches = sorted(
                    matches.items(), key=lambda x: x[1]["score"], reverse=True
                )[:TOP_N_MATCHES]

                match_strs = []
                for ap_lineid, info in sorted_matches:
                    relationship = info["relationship"]
                    score = info["score"]
                    match_strs.append(f"{ap_lineid} ({relationship}, {score:.2f})")

                full_match_str = "; ".join(match_strs)

                rag_idx = df[df["LineID"] == rag_lineid].index[0]
                df.at[rag_idx, col_name] = full_match_str

            # Progress update
            if processed % 5 == 0 or processed == total_rag:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = total_rag - processed
                eta = remaining / rate if rate > 0 else 0

                print(
                    f"    Progress: {processed}/{total_rag} RAG actions ({processed / total_rag * 100:.1f}%) - {rate:.1f} actions/sec - ETA: {eta / 60:.1f}min"
                )

            time.sleep(BATCH_PAUSE_SECONDS)

    else:
        # Fallback: TF-IDF
        print("    Using TF-IDF fallback...")
        all_texts = list(rag_lookup.values()) + list(ap_lookup.values())

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        rag_size = len(rag_lookup)
        rag_vectors = tfidf_matrix[:rag_size]
        ap_vectors = tfidf_matrix[rag_size:]

        similarity_matrix = cosine_similarity(rag_vectors, ap_vectors)

        for i, rag_lineid in enumerate(rag_lookup.keys()):
            similarities = similarity_matrix[i]

            top_indices = np.argsort(similarities)[::-1][:TOP_N_MATCHES]
            top_scores = similarities[top_indices]

            matches = []
            for idx, score in zip(top_indices, top_scores):
                if score >= FUZZY_MIN_SCORE:
                    ap_lineid = list(ap_lookup.keys())[idx]
                    matches.append(f"{ap_lineid} (Related, {score:.2f})")

            if matches:
                full_match_str = "; ".join(matches)
                rag_idx = df[df["LineID"] == rag_lineid].index[0]
                df.at[rag_idx, col_name] = full_match_str

    return df


def main():
    """Main workflow with checkpoint support."""
    print("=" * 70)
    print("RAG ALIGNMENT WORKFLOW - UNPAIRED VERSION WITH CHECKPOINTS")
    print("=" * 70)

    print(
        f"\nCategorisation: {'NLTK + Keyword Taxonomy' if NLTK_AVAILABLE else 'Disabled'}"
    )
    print("Within-period matching: Fast TF-IDF (actions expected to be identical)")
    if GEMINI_AVAILABLE:
        print("Cross-period matching: Gemini AI (intelligent relationship analysis)")
    elif EMBEDDINGS_AVAILABLE:
        print("Cross-period matching: Gemini Embeddings (semantic)")
    else:
        print("Cross-period matching: TF-IDF fallback")

    # Check for existing checkpoint
    print("\n" + "=" * 70)
    print("CHECKING FOR CHECKPOINTS")
    print("=" * 70)

    checkpoint_df, checkpoint_state = load_checkpoint()
    resume_from_checkpoint = False

    if checkpoint_df is not None and checkpoint_state is not None:
        response = (
            input("\nFound existing checkpoint. Resume from it? (y/n): ")
            .strip()
            .lower()
        )
        if response == "y":
            resume_from_checkpoint = True
            df = checkpoint_df
        else:
            clear_response = (
                input("Clear checkpoints and start fresh? (y/n): ").strip().lower()
            )
            if clear_response == "y":
                clear_checkpoints()

    # Load data (if not resuming)
    if not resume_from_checkpoint:
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)

        try:
            df = pd.read_csv(INPUT_CSV_PATH)
            print(f"✓ Loaded {len(df)} rows × {len(df.columns)} columns")
        except FileNotFoundError:
            print(f"ERROR: Input file not found: {INPUT_CSV_PATH}")
            return
        except Exception as e:
            print(f"ERROR loading CSV: {e}")
            return

        # Parse document types
        print("\nParsing document types...")
        df["Doc_Type"] = None
        df["Period"] = None

        for idx in df.index:
            doc_type = df.at[idx, "Document Type"]
            dtype, period = parse_document_type(doc_type)
            df.at[idx, "Doc_Type"] = dtype
            df.at[idx, "Period"] = period

    # Get dataset structure
    institutions = df["Institution"].dropna().unique()
    periods = sorted([p for p in df["Period"].dropna().unique() if pd.notna(p)])

    # Display structure
    print("\nDataset Structure:")
    print(f"  Institutions: {', '.join(map(str, institutions))}")
    print(f"  Periods: {[int(p) for p in periods]}")

    for inst in institutions:
        print(f"\n  {inst}:")
        inst_df = df[df["Institution"] == inst]
        for period in periods:
            ap_count = len(inst_df[inst_df["Document Type"] == f"AP{int(period)}"])
            rag_count = len(inst_df[inst_df["Document Type"] == f"RAG_AP{int(period)}"])
            if ap_count > 0 or rag_count > 0:
                print(f"    Period {int(period)}: {ap_count} AP, {rag_count} RAG_AP")

    # Determine starting point
    if resume_from_checkpoint:
        start_task = checkpoint_state.get("task", "categorisation")
        last_institution = checkpoint_state.get("last_institution")
        last_period_from = checkpoint_state.get("last_period_from")
        last_period_to = checkpoint_state.get("last_period_to")

        print(f"\n✓ Resuming from: {start_task}")
        if last_institution:
            print(
                f"  Last completed: {last_institution}, Period {last_period_from}→{last_period_to}"
            )
    else:
        start_task = "categorisation"
        last_institution = None
        last_period_from = None
        last_period_to = None

    # Load keyword taxonomy (if needed for categorisation)
    category_keywords = {}
    if NLTK_AVAILABLE and start_task == "categorisation":
        try:
            category_keywords = load_and_preprocess_keywords(
                KEYWORD_JSON_PATH, PROCESSING_MODE
            )
            print(f"\n✓ Loaded {len(category_keywords)} categories from taxonomy")
        except FileNotFoundError:
            print(f"\nWARNING: Keyword file not found: {KEYWORD_JSON_PATH}")
        except Exception as e:
            print(f"\nWARNING: Error loading keywords: {e}")

    # Task 1: Categorisation
    if start_task == "categorisation":
        print("\n" + "=" * 70)
        print("TASK 1: CATEGORISING ACTIONS")
        print("=" * 70)

        if NLTK_AVAILABLE and category_keywords:
            if "Category" not in df.columns:
                df["Category"] = pd.Series(dtype="string")

            categorised_count = 0
            for idx in df.index:
                action_text = df.at[idx, "Action"]
                if (
                    pd.notna(action_text)
                    and isinstance(action_text, str)
                    and action_text.strip()
                ):
                    categories = categorise_action_with_scoring(
                        action_text, category_keywords, PROCESSING_MODE
                    )
                    if categories:
                        df.at[idx, "Category"] = "; ".join(categories)
                        categorised_count += 1

            print(f"✓ Categorised {categorised_count} actions")
        else:
            print("  ⚠ Skipping categorisation")

        start_task = "within_period"

    # Task 2: Within-period matching
    if start_task in ["categorisation", "within_period"]:
        print("\n" + "=" * 70)
        print("TASK 2: WITHIN-PERIOD MATCHING (AP vs RAG_AP)")
        print("Using fast TF-IDF (expected to be mostly identical)")
        print("=" * 70)

        if "Match AP vs RAG_AP (LineID, Status)" not in df.columns:
            df["Match AP vs RAG_AP (LineID, Status)"] = pd.Series(dtype="string")
        if "Match RAG_AP vs AP (LineID, Status)" not in df.columns:
            df["Match RAG_AP vs AP (LineID, Status)"] = pd.Series(dtype="string")

        for inst in institutions:
            for period in periods:
                df = match_within_period_simple(df, inst, int(period))

        print("\n✓ Within-period matching complete")

        # Save checkpoint after Task 2
        state_info = {
            "task": "cross_period",
            "last_completed": "Task 2: Within-period matching",
            "institutions": list(institutions),
            "periods": [int(p) for p in periods],
        }
        save_checkpoint(df, state_info, "after_task2")

        start_task = "cross_period"

    # Task 3: Cross-period matching
    if start_task == "cross_period":
        print("\n" + "=" * 70)
        print("TASK 3: CROSS-PERIOD MATCHING (RAG_AP(x) → AP(x+1))")
        if GEMINI_AVAILABLE:
            print("Using Gemini AI for intelligent relationship analysis")
        else:
            print("Using TF-IDF fallback")
        print("=" * 70)

        # Create columns for each period transition
        for i in range(len(periods) - 1):
            period_from = int(periods[i])
            period_to = int(periods[i + 1])
            col_name = f"Fuzzy Match RAG_AP{period_from} → AP{period_to} (LineID, Relationship)"
            if col_name not in df.columns:
                df[col_name] = pd.Series(dtype="string")

        # Count total transitions to process
        total_transitions = 0
        transitions_list = []

        for inst in institutions:
            for i in range(len(periods) - 1):
                period_from = int(periods[i])
                period_to = int(periods[i + 1])
                inst_mask = df["Institution"] == inst
                rag_mask = inst_mask & (df["Document Type"] == f"RAG_AP{period_from}")
                rag_count = len(df[rag_mask])
                if rag_count > 0:
                    transitions_list.append((inst, period_from, period_to))
                    total_transitions += 1

        # Determine starting point if resuming
        start_index = 0
        if (
            resume_from_checkpoint
            and last_institution
            and last_period_from
            and last_period_to
        ):
            try:
                last_transition = (last_institution, last_period_from, last_period_to)
                if last_transition in transitions_list:
                    start_index = transitions_list.index(last_transition) + 1
                    print(
                        f"\n✓ Resuming from transition {start_index + 1}/{total_transitions}"
                    )
            except ValueError:
                print(
                    "\n  Warning: Could not find last checkpoint in transitions list, starting from beginning"
                )

        print(
            f"\nProcessing {total_transitions - start_index} remaining transitions..."
        )
        transition_num = start_index
        overall_start = time.time()

        for inst, period_from, period_to in transitions_list[start_index:]:
            transition_num += 1
            print(
                f"\n[Transition {transition_num}/{total_transitions}] {inst}: Period {period_from} → {period_to}"
            )

            df = match_cross_period_intelligent(df, inst, period_from, period_to)

            # Save checkpoint after each transition
            elapsed = time.time() - overall_start
            avg_per_transition = (
                elapsed / (transition_num - start_index)
                if (transition_num - start_index) > 0
                else 0
            )
            remaining_transitions = total_transitions - transition_num
            estimated_remaining = avg_per_transition * remaining_transitions

            print(
                f"  Overall progress: {transition_num}/{total_transitions} transitions complete"
            )
            print(
                f"  Time: {elapsed / 60:.1f}min elapsed, ~{estimated_remaining / 60:.1f}min remaining"
            )

            # Save checkpoint
            state_info = {
                "task": "cross_period",
                "last_completed": f"{inst}: Period {period_from}→{period_to}",
                "last_institution": inst,
                "last_period_from": period_from,
                "last_period_to": period_to,
                "transition_num": transition_num,
                "total_transitions": total_transitions,
                "institutions": list(institutions),
                "periods": [int(p) for p in periods],
            }
            save_checkpoint(df, state_info, f"transition_{transition_num}")

        print("\n✓ Cross-period matching complete")

    # Calculate total cross-period matching statistics
    total_comparisons = 0
    for inst in institutions:
        for i in range(len(periods) - 1):
            period_from = int(periods[i])
            period_to = int(periods[i + 1])
            col_name = f"Fuzzy Match RAG_AP{period_from} → AP{period_to} (LineID, Relationship)"
            if col_name in df.columns:
                total_comparisons += df[col_name].notna().sum()

    print(f"  Total cross-period comparisons completed: {total_comparisons}")

    # Report statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)

    if "Match AP vs RAG_AP (LineID, Status)" in df.columns:
        within_matched = df["Match AP vs RAG_AP (LineID, Status)"].notna().sum()
        print(f"\n  Within-period matches (AP): {within_matched}")

    if GEMINI_AVAILABLE and len(GEMINI_RAW_RESPONSES) > 0:
        print(f"  Gemini API calls (cross-period): {len(GEMINI_RAW_RESPONSES)}")

        gemini_debug = {
            "summary": {
                "total_responses": len(GEMINI_RAW_RESPONSES),
                "model_used": GEMINI_MODEL,
            },
            "responses": GEMINI_RAW_RESPONSES,
        }

        debug_file = OUTPUT_CSV_PATH.replace(".csv", "_gemini_debug.json")
        try:
            with open(debug_file, "w") as f:
                json.dump(gemini_debug, f, indent=2)
            print(f"  Debug: Saved Gemini responses to {debug_file}")
        except Exception as e:
            print(f"  Warning: Could not save debug file: {e}")

    # Save final results
    print("\n" + "=" * 70)
    print("SAVING FINAL RESULTS")
    print("=" * 70)

    try:
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"✓ Results saved to: {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"ERROR saving results: {e}")
        return

    # Clear checkpoints on successful completion
    clear_checkpoints()

    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
