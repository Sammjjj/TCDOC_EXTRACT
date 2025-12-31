# API Reference

## Overview

This document provides detailed technical documentation for all modules in the TCDOC_EXTRACT toolkit.

---

## Table of Contents

1. [document_extract2.py](#document_extract2py) - Document text extraction
2. [predefined_categories6.py](#predefined_categories6py) - Action categorization
3. [evaluate_multi_col.py](#evaluate_multi_colpy) - Performance evaluation
4. [subclus_calc_weight.py](#subclus_calc_weightpy) - TF-IDF keyword analysis
5. [rule_based_class.py](#rule_based_classpy) - Sub-category classification
6. [rag_extract_simple12.py](#rag_extract_simple12py) - RAG document extraction
7. [rag_alignment_workflow_unpaired8.py](#rag_alignment_workflow_unpaired8py) - Action alignment workflow
8. [enumerate_resources.py](#enumerate_resourcespy) - Resource enumeration
9. [create_graphical_summary_multi_institution.py](#create_graphical_summary_multi_institutionpy) - Graphical visualization
10. [create_comprehensive_summary_multi_institution.py](#create_comprehensive_summary_multi_institutionpy) - Text-based summary tables

---

## document_extract2.py

### Purpose
Extracts action items from Technician Commitment documents stored in Google Drive using AI-powered text analysis.

### Key Functions

#### `authenticate()`
Handles OAuth2 authentication for Google Drive API access.

**Returns:**
- `Credentials`: Valid Google OAuth2 credentials

**Files Used:**
- `credentials.json` - OAuth2 client credentials
- `token.json` - Stored authentication token

#### `get_files_from_folder(service, folder_id)`
Retrieves list of files from a specified Google Drive folder.

**Parameters:**
- `service`: Google Drive API service instance
- `folder_id` (str): Google Drive folder ID

**Returns:**
- `list`: List of file metadata dictionaries

#### `get_text_from_file(service, file_id, mime_type)`
Extracts text content from Google Docs or PDF files.

**Parameters:**
- `service`: Google Drive API service instance
- `file_id` (str): Google Drive file ID
- `mime_type` (str): MIME type of the file

**Returns:**
- `str`: Extracted text content

**Supported Formats:**
- Google Docs (`application/vnd.google-apps.document`)
- PDF files (`application/pdf`)

#### `extract_actions_with_gemini(text)`
Uses Google Gemini AI to extract discrete action items from document text.

**Parameters:**
- `text` (str): Full document text

**Returns:**
- `list`: List of extracted action strings

**Configuration:**
- Model: `gemini-2.5-pro`
- Temperature: 0.0 (deterministic output)
- Location: `us-central1`

### Configuration Variables

```python
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'
LOCATION = 'us-central1'
OUTPUT_CSV_FILE = 'actions_output2.csv'
PROJECT_ID = 'tcdocext'
MODEL_ID = 'gemini-2.5-pro'
```

### Output Format

CSV file with columns:
- `Document Name`: Source document filename
- `Extracted Action`: Individual action item text

---

## predefined_categories6.py

### Purpose
Categorizes extracted actions into predefined thematic categories using weighted keyword matching and NLP preprocessing.

### Key Functions

#### `preprocess_text(text, mode='lemmatization')`
Tokenizes and normalizes text using NLTK.

**Parameters:**
- `text` (str): Input text to preprocess
- `mode` (str): Processing mode ('lemmatization' or basic tokenization)

**Returns:**
- `list`: List of preprocessed tokens

**Processing Steps:**
1. Tokenization
2. Lowercase conversion
3. Stopword removal (with custom exceptions)
4. POS tagging
5. Lemmatization

#### `get_wordnet_pos(treebank_tag)`
Converts Penn Treebank POS tags to WordNet tags.

**Parameters:**
- `treebank_tag` (str): Penn Treebank POS tag

**Returns:**
- WordNet POS constant

#### `load_and_preprocess_keywords(filename, mode='lemmatization')`
Loads and preprocesses keyword dictionary from JSON.

**Parameters:**
- `filename` (str): Path to keyword JSON file
- `mode` (str): Preprocessing mode

**Returns:**
- `dict`: Category -> [(token_tuple, weight), ...] mapping

#### `categorize_action_with_scoring(action_text, category_keywords, mode='lemmatization', use_ratio_threshold=False, ratio=0.3, min_score=0.1)`
Scores action against all categories using weighted keyword matching.

**Parameters:**
- `action_text` (str): Action text to categorize
- `category_keywords` (dict): Preprocessed keyword dictionary
- `mode` (str): Processing mode
- `use_ratio_threshold` (bool): Whether to use relative threshold
- `ratio` (float): Relative score threshold (0.0-1.0)
- `min_score` (float): Minimum absolute score threshold

**Returns:**
- `dict`: Category -> score mapping

### Configuration Variables

```python
PROCESSING_MODE = "lemmatization"
INPUT_CSV_PATH = 'All_RAG.csv'
KEYWORD_JSON_PATH = 'keyword_categories_4.json'
OUTPUT_CSV_PATH = 'categorized_actions_output_RAG.csv'
MIN_WORD_COUNT = 3
ACTION_COLUMN_NAME = 'Action'
```

### Stopword Exceptions

Custom stopwords retained for domain-specific meaning:
- `of`, `on`, `for`, `with`, `against`, `up`
- `ongoing`, `regular`, `provide`

### Output Format

CSV with one-hot encoded category columns:
- Original action columns preserved
- Binary column per category (1 = assigned, 0 = not assigned)
- `Uncategorised` column for unmatched actions

---

## evaluate_multi_col.py

### Purpose
Evaluates categorization performance against ground truth using confusion matrix metrics.

### Key Functions

#### `normalize_text(s)`
Normalizes text for comparison.

**Parameters:**
- `s` (str): Text to normalize

**Returns:**
- `str`: Normalized text (lowercase, collapsed whitespace)

#### `calculate_metrics(df_merged, category_cols)`
Calculates sensitivity and specificity for each category.

**Parameters:**
- `df_merged` (DataFrame): Merged predictions and ground truth
- `category_cols` (list): List of category column names

**Returns:**
- `DataFrame`: Performance metrics per category

**Metrics Calculated:**
- **Sensitivity (Recall)**: TP / (TP + FN)
- **Specificity**: TN / (TN + FP)
- True Positives, False Positives, True Negatives, False Negatives
- Support (actual positive count)

### Configuration Variables

```python
PREDICTIONS_FILE = 'categorized_actions_output_multicol_weighted2.csv'
GROUND_TRUTH_FILE = 'ground_truth2.csv'
OUTPUT_REPORT_FILE = 'sensitivity_specificity_report_weighted2.csv'
PREDICTION_ACTION_COL = 'Extracted Action'
GROUND_TRUTH_ACTION_COL = 'Extracted Action'
GROUND_TRUTH_CATEGORY_COL = 'Ground Truth'
```

### Output Format

CSV with columns:
- `Category`: Category name
- `Sensitivity (Recall)`: 0.0-1.0
- `Specificity`: 0.0-1.0
- `True Positives`, `False Positives`, `True Negatives`, `False Negatives`
- `Support (Actual Positives)`

---

## subclus_calc_weight.py

### Purpose
Identifies salient keywords within each category using TF-IDF analysis to inform sub-category creation.

### Key Functions

#### Main Processing Loop
Iterates through each category and:
1. Filters actions for category
2. Applies TF-IDF vectorization
3. Sums scores across documents
4. Extracts top 15 keywords
5. Writes to output file

### TF-IDF Configuration

```python
TfidfVectorizer(
    stop_words='english',
    max_features=1000,
    smooth_idf=True,
    sublinear_tf=True
)
```

### Input/Output

**Input:**
- `Actions_FinalData - All_Data.csv` with columns:
  - `Categories`: Category name
  - `Extracted Action`: Action text

**Output:**
- `initial_keyword_analysis.txt`: Top 15 keywords per category

---

## rule_based_class.py

### Purpose
Assigns actions to sub-categories within their main category using predefined keyword rules.

### Key Data Structure

#### `category_to_subclusters`
Dictionary mapping:
```python
{
    'Category Name': {
        'Sub-Category Name': ['keyword1', 'keyword2', ...],
        ...
    },
    ...
}
```

**Coverage:**
- 17 main categories
- 3 sub-categories per main category
- Manually curated keyword lists

### Key Functions

#### `assign_subcluster_name(statement, subcluster_definitions)`
Assigns statement to best-matching sub-category.

**Parameters:**
- `statement` (str): Action text
- `subcluster_definitions` (dict): Sub-category -> keyword list mapping

**Returns:**
- `str`: Assigned sub-category name or 'Unassigned'

**Algorithm:**
1. Convert statement to lowercase
2. For each sub-category, count whole-word keyword matches
3. Assign to sub-category with highest count
4. Return 'Unassigned' if all scores are 0

### Input/Output

**Input:**
- `Actions_FinalData - All_Data.csv`

**Output:**
- `named_subcluster_analysis.csv` with added column:
  - `predefined_subcluster`: Sub-category assignment

---

## rag_extract_simple12.py

### Purpose
Extracts actions with RAG (Red/Amber/Green) status indicators from assessment documents.

### Key Functions

#### `authenticate()`
Same OAuth2 authentication as document_extract2.py.

#### `parse_filename_metadata(filename)`
Extracts institution, year, document type, and RAG flag from filename.

**Parameters:**
- `filename` (str): Document filename

**Returns:**
- `dict`: Metadata with keys `institution`, `year`, `doc_type`, `is_rag`

**Supported Patterns:**
- `YYYY-Institution-DocType`
- `Institution-YYYY-DocType`
- `InstitutionActionPlanYYYY`
- `InstitutionRAGYYYY`

#### Structured Table Extraction Algorithm
Lines 223-517 implement intelligent table parsing:

1. **Header Detection**: Scans rows 0-1 for "action" and "RAG" columns
2. **Column Selection**: Prioritizes exact "action" match, then most content
3. **Continuation Tables**: Tracks columns across headerless continuation tables
4. **Multi-Row Merging**: Combines split actions using capitalization heuristics
5. **RAG Extraction**: Identifies R/A/G markers and normalizes to single-letter format

#### Unstructured Text Extraction
Lines 520-574 use Gemini AI for documents without tables:

**Model:** `gemini-2.0-flash-001`

**Prompt Instructions:**
1. Identify discrete actions as complete sentences
2. Remove RAG indicators from action text
3. Preserve sentence completeness
4. Clean formatting artifacts

**Input Limit:** First 50,000 characters per document

### Configuration Variables

```python
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"
LOCATION = "us-central1"
OUTPUT_CSV_FILE = "rag_actions_extraction.csv"
PROJECT_ID = "tcdocprog"
MODEL_ID = "gemini-2.0-flash"
```

### Output Format

CSV with columns:
- `LineID`: Sequential identifier
- `Institution`: Organization name
- `Document Type`: AP1, AP2, AP3, RAG_AP1, RAG_AP2, RAG_AP3
- `Action`: Full action text
- `RAG value`: R, A, or G (empty for Action Plans)
- `Category`: Populated by categorization step

---

## rag_alignment_workflow_unpaired8.py

### Purpose
Comprehensive workflow for matching actions across documents and time periods, with checkpoint-based resumability.

### Key Functions

#### Task 1: Action Categorization
Applies same categorization as `predefined_categories6.py` to all extracted actions.

#### Task 2: Within-Period TF-IDF Matching

**Function:** Tier 1 matching (lines 117-289)

**Algorithm:**
1. Filter actions to same period (e.g., AP1 vs RAG_AP1)
2. Compute TF-IDF vectors
3. Calculate cosine similarity
4. Classify matches:
   - `Identical`: similarity ≥ 0.95
   - `Match`: similarity ≥ 0.85
   - `Similar`: similarity ≥ 0.70

**Bidirectional:** AP→RAG_AP and RAG_AP→AP

#### Task 3: Cross-Period Gemini AI Matching

**Function:** Tier 2 matching (lines 289-383)

**Algorithm:**
1. Compare RAG_AP(n) actions against AP(n+1) actions
2. Batch up to 10 comparisons per API call
3. Include category metadata in prompt
4. Request relationship classification:
   - `Identical`: Unchanged
   - `Extended`: More ambitious
   - `Narrowed`: More focused
   - `Related`: Connected but different
   - `Unrelated`: No connection
5. Return JSON with relationship and confidence score (0.0-1.0)
6. Filter to score ≥ 0.5, keep top 5 matches

#### Rate Limiting (lines 177-196)

**Implementation:**
- Sliding window deque tracking request timestamps
- 15 requests per minute limit
- Automatic sleep when approaching limit
- 4-second pause between batches

#### Checkpoint System (lines 94-175)

**Features:**
- Saves after each major task completion
- Saves after each institution-period transition
- Stores dataframe to `checkpoints/checkpoint_name.csv`
- Stores metadata to `checkpoints/workflow_state.json`
- Auto-detects and offers resume on restart

**Metadata Tracked:**
- Last completed institution
- Last completed period transition
- Progress metrics
- Timestamp

#### Trajectory Classification

**Codes:**
- `C` (Continued): Same focus (Identical/Extended)
- `R` (Related): Different focus (Related/Narrowed)
- `S` (Stopped): No successor
- `N` (New): No predecessor
- `U` (Unknown): Insufficient data

#### Derived Metrics

Seven quantitative metrics per category:

1. **Growth Rate**: (Period 3 count - Period 1 count) / Period 1 × 100%
2. **Continuation Rate**: (Continued + Related) / (Continued + Related + Stopped) × 100%
3. **Trajectory Diversity**: Shannon entropy of trajectory distribution (0-100%)
4. **Splitting Ratio**: Avg successors per source action
5. **Merging Ratio**: Avg predecessors per target action
6. **Longevity Score**: % of Period 1 actions traceable to Period 3
7. **Category Coherence**: 1 - std dev of trajectory types

### Configuration Variables

```python
PROJECT_ID = "tcdocprog"
LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
INPUT_CSV = "rag_actions_extraction.csv"
OUTPUT_CSV = "rag_actions_extraction_MATCHED.csv"
KEYWORD_JSON = "keyword_categories_4.json"
CHECKPOINT_DIR = "checkpoints"
```

### Output Format

Enriched CSV with:
- All original extraction columns
- Within-period match columns (bidirectional)
- Cross-period match columns with relationships
- Trajectory codes
- Derived metrics

---

## enumerate_resources.py

### Purpose
Searches action plans for mentions of external resources, professional bodies, and support structures.

### Key Functions

#### `load_keyphrases_from_json(filename)`
Loads resource names from JSON dictionary.

**Parameters:**
- `filename` (str): Path to resources JSON

**Returns:**
- `list`: Flat list of resource names/phrases

#### `enumerate_exact_matches(actions_df, key_phrases)`
Searches for whole-word phrase matches with context display.

**Parameters:**
- `actions_df` (DataFrame): Actions dataset
- `key_phrases` (list): Resource names to search for

**Returns:**
- `DataFrame`: Phrase counts

**Features:**
- Whole-word boundary matching (`\b...\b`)
- Case-insensitive search
- Context display (10 words before/after)
- Terminal highlighting with ANSI colors
- Multiple matches per action counted

### Configuration Variables

```python
ACTIONS_CSV_PATH = 'actions_output2_all.csv'
KEYPHRASES_JSON_PATH = 'external_resources.json'
ACTION_COLUMN_NAME = 'Extracted Action'
OUTPUT_CSV_PATH = 'exact_phrase_counts_whole_word.csv'
CONTEXT_WORD_COUNT = 10
```

### Terminal Output

Colored context display:
```
Found: [phrase] in row 42 >> ...context before PHRASE context after...
```

### Output Format

CSV with columns:
- `Key Phrase`: Resource name
- `Count`: Number of whole-word matches

Sorted by count (descending)

---

## create_graphical_summary_multi_institution.py

### Purpose
Generates publication-quality visualizations of category-level action evolution across assessment periods.

### Key Features

#### Multi-Dimensional Encoding
Represents 5 data dimensions simultaneously:
1. **Temporal period**: Horizontal axis
2. **Category membership**: Vertical axis
3. **Action volume**: Node size
4. **Trajectory type**: Edge color
5. **Performance metrics**: Heatmap overlay

#### Standardized Color Scheme

```python
COLOURS = {
    "Continued": "#27ae60",    # Green
    "Extended": "#27ae60",     # Green
    "Related": "#f39c12",      # Orange
    "Narrowed": "#f39c12",     # Orange
    "Stopped": "#e74c3c",      # Red
    "New": "#3498db",          # Blue
    "Unknown": "#95a5a6",      # Grey
}
```

### Output

- One PNG per institution
- Filename: `{institution}_category_summary.png`
- High-resolution suitable for publication

---

## create_comprehensive_summary_multi_institution.py

### Purpose
Generates text-based summary tables for all institutions in a single output file.

### Key Functions

#### `parse_fuzzy_match(match_text)`
Parses Gemini match results.

**Parameters:**
- `match_text` (str): Fuzzy match column value

**Returns:**
- `list`: [(lineid, relationship, score), ...]

**Format:** `"LineID (Relationship, Score); LineID (Relationship, Score); ..."`

### Output Tables Per Institution

1. **Forward-looking transitions**: Where actions go in next period
2. **Backward-looking origins**: Where actions came from
3. **Derived metrics**: All 7 quantitative metrics

### Output Format

Single text file with sections per institution, formatted as readable tables.

---

## Data Flow Diagram

```
Google Drive Documents
        ↓
document_extract2.py → actions_output2.csv
        ↓
predefined_categories6.py → categorized_actions.csv
        ↓
evaluate_multi_col.py → performance_report.csv

subclus_calc_weight.py → keyword_analysis.txt
        ↓
rule_based_class.py → subclustered_actions.csv

Google Drive RAG Docs
        ↓
rag_extract_simple12.py → rag_actions_extraction.csv
        ↓
rag_alignment_workflow_unpaired8.py → rag_actions_extraction_MATCHED.csv
        ↓
        ├─→ create_graphical_summary_multi_institution.py → PNG files
        └─→ create_comprehensive_summary_multi_institution.py → text tables

enumerate_resources.py → resource_counts.csv
```

---

## Error Handling

### Common Issues

#### Authentication Errors
- **Cause**: Missing or invalid `credentials.json`
- **Solution**: Download OAuth2 credentials from Google Cloud Console

#### NLTK Data Not Found
- **Cause**: Missing NLTK corpora
- **Solution**: Run `python -m nltk.downloader punkt averaged_perceptron_tagger wordnet stopwords`

#### Vertex AI Quota Exceeded
- **Cause**: Free tier API limits reached
- **Solution**: Workflow automatically falls back to TF-IDF; or upgrade GCP account

#### File Not Found
- **Cause**: Input CSV from previous step missing
- **Solution**: Run pipeline steps in sequence

---

## Performance Considerations

### Optimization Tips

1. **Batch Processing**: `rag_alignment_workflow_unpaired8.py` batches 10 comparisons per API call
2. **Checkpointing**: Use resume functionality for large datasets
3. **Rate Limiting**: Conservative 15 req/min prevents quota exhaustion
4. **TF-IDF Caching**: Within-period matching is fast and cached
5. **Multiprocessing**: Not currently implemented but possible for independent institutions

### Typical Processing Times

- **document_extract2.py**: ~10 sec per document
- **predefined_categories6.py**: ~1 sec per 1000 actions
- **rag_alignment_workflow_unpaired8.py**: ~2-3 hours for full dataset (with checkpoints)
- **Visualization scripts**: ~30 sec per institution

---

## Version History

- **v1.0** (August 2025): Initial release for Technician Commitment analysis
- Associated paper: "Synergies and Gaps in Technical Skills Development in UK Universities"
- Author: Dr. Samuel J Jackson

---

## Support

For issues or questions:
1. Check configuration variables match your file paths
2. Verify all dependencies installed (`pip install -r requirements.txt`)
3. Review error messages for specific file/column names
4. Consult Methods.txt for methodological details
