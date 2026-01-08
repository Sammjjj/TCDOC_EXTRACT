# TCDOC_EXTRACT

**Automated Analysis Toolkit for Technician Commitment Documents**

A comprehensive Python-based natural language processing (NLP) toolkit for extracting, categorizing, and analyzing action items from institutional commitment documents. Developed for analysis of UK universities' Technician Commitment action plans and progress reports.

Zenodo repository accompanying this code/publication: https://zenodo.org/records/18185138
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

This toolkit accompanies the paper:

> **"Synergies and Gaps in Support for Technical Career Development in UK Higher Education and Research: A Semi-Quantitative Analysis of ‘Technician Commitment’ Action Plans and Progress Reports using Natural Language Processing"**
>
> Dr. Samuel J Jackson, August 2025

TCDOC_EXTRACT provides a complete workflow for:
- 📄 **Extracting** action items from PDF and Google Doc formats
- 🏷️ **Categorizing** actions using weighted keyword matching
- 📊 **Analyzing** institutional progress through RAG (Red/Amber/Green) assessments
- 🔗 **Tracking** action evolution across multiple time periods
- 📈 **Visualizing** institutional commitment patterns and trends

---

## Key Features

### 🤖 AI-Powered Extraction
- Google Gemini 2.5 language model for intelligent action extraction
- Handles diverse document formats (PDF, Google Docs, Word)
- Preserves semantic completeness of actions

### 🎯 Multi-Level Categorization
- 26 predefined thematic categories
- Weighted keyword lexicon with 500+ domain-specific terms
- Sub-category classification for nuanced analysis
- NLTK-based lemmatization and POS tagging

### 📊 Longitudinal Tracking
- Three-tier action matching strategy (TF-IDF + AI semantic analysis)
- Trajectory classification (Continued, Related, Stopped, New)
- Cross-period relationship detection (Identical, Extended, Narrowed)
- 7 derived institutional metrics

### ⚡ Production-Ready
- Checkpoint-based resumability for large datasets
- Automatic rate limiting and quota management
- Batch processing optimization (10x API cost reduction)
- Comprehensive error handling

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Sammjjj/TCDOC_EXTRACT.git
cd TCDOC_EXTRACT

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet stopwords
```

See [INSTALL.md](INSTALL.md) for detailed setup including Google Cloud configuration.

### Basic Usage

#### 1. Extract Actions from Documents

```bash
python document_extract2.py
```

**Output:** `actions_output2.csv` - Extracted action items

#### 2. Categorize Actions

```bash
python predefined_categories6.py
```

**Input:** `actions_output2.csv`
**Output:** `categorized_actions_output.csv` - Actions with category assignments

#### 3. Analyze RAG Assessments

```bash
# Extract RAG data
python rag_extract_simple12.py

# Align actions across periods
python rag_alignment_workflow_unpaired8.py

# Generate visualizations
python create_graphical_summary_multi_institution.py
```

**Output:** Matched actions with trajectory analysis and PNG visualizations

---

## Documentation

| Document | Description |
|----------|-------------|
| [INSTALL.md](INSTALL.md) | Complete installation guide with Google Cloud setup |
| [API_REFERENCE.md](API_REFERENCE.md) | Detailed technical documentation for all modules |
| [Methods.txt](Methods.txt) | Methodological background and analytical approach |
| [LICENSE](LICENSE) | Apache 2.0 license terms |

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION                         │
│                                                             │
│  Google Drive ──→ document_extract2.py                      │
│                   rag_extract_simple12.py                   │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   CATEGORIZATION                            │
│                                                             │
│  predefined_categories6.py ──→ 26 Categories                │
│  rule_based_class.py ──→ Sub-categories                     │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 LONGITUDINAL ANALYSIS                       │
│                                                             │
│  rag_alignment_workflow_unpaired8.py:                       │
│    • Within-period matching (TF-IDF)                        │
│    • Cross-period matching (Gemini AI)                      │
│    • Trajectory classification                              │
│    • Derived metrics                                        │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│               VISUALIZATION & REPORTING                     │
│                                                             │
│  create_graphical_summary_multi_institution.py              │
│  create_comprehensive_summary_multi_institution.py          │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Reference

### Core Analysis Pipeline

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `document_extract2.py` | Extract actions from documents | Google Drive folder | `actions_output2.csv` |
| `predefined_categories6.py` | Categorize actions | Action CSV | `categorized_actions.csv` |
| `evaluate_multi_col.py` | Evaluate categorization performance | Predictions + ground truth | Performance metrics |

### Sub-Category Analysis

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `subclus_calc_weight.py` | TF-IDF keyword analysis | Categorized actions | `keyword_analysis.txt` |
| `rule_based_class.py` | Assign sub-categories | Categorized actions | Actions with sub-categories |

### RAG Assessment Analysis

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `rag_extract_simple12.py` | Extract RAG assessments | RAG documents | `rag_actions_extraction.csv` |
| `rag_alignment_workflow_unpaired8.py` | Match actions across periods | RAG extraction | `rag_actions_extraction_MATCHED.csv` |

### Visualization

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `create_graphical_summary_multi_institution.py` | Generate category flow diagrams | Matched actions | PNG visualizations |
| `create_comprehensive_summary_multi_institution.py` | Create text-based summary tables | Matched actions | Text report |

### Utilities

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `enumerate_resources.py` | Find resource mentions | Actions + resource list | Resource counts |

---

## Configuration Files

### keyword_categories_4.json
Defines the 26-category taxonomy with weighted keyword lists.

**Structure:**
```json
{
  "Category Name": {
    "keyword phrase": weight,
    "another phrase": weight
  }
}
```

**Categories:**
- Apprenticeships, Internships, Placements
- Career Frameworks & Role Definition
- Career Pathways & Progression
- Data & Workforce Analysis
- EDI (Equality, Diversity & Inclusion)
- External Collaboration & Partnerships
- Funding for Technicians
- Mentorship & Support
- Monitoring & Evaluation of TC
- Networking and Presenting
- Ongoing Visibility & Communication
- Professional Registration & Accreditation
- Recognition & Awards
- Recruitment & Onboarding
- Representation in Institutional Governance
- Technician Leadership
- Technician Voice & Feedback
- Training & Skills Development
- ...and 8 more

### credentials.json
OAuth2 credentials for Google Drive API access. **Not included** - you must create this via Google Cloud Console.

See [INSTALL.md](INSTALL.md) for setup instructions.

---

## Example Outputs

### Categorized Actions CSV

```csv
Document Name,Extracted Action,Training & Skills Development,Career Pathways & Progression,Recognition & Awards,Uncategorised
Bristol,Provide access to professional development workshops,1,0,0,0
Oxford,Implement a technician career progression framework,0,1,0,0
Cambridge,Establish an annual technician excellence award,0,0,1,0
```

### RAG Matched Actions CSV

```csv
LineID,Institution,Period,Action,RAG value,Category,Match AP vs RAG_AP,Match RAG_AP vs AP2,Trajectory
1,Bristol,1,Support professional registration,G,Professional Registration,2 (Identical; 0.98),15 (Extended; 0.85),Continued
```

### Derived Metrics

- **Growth Rate**: 45.2% (Period 1 → Period 3)
- **Continuation Rate**: 78.3% (actions sustained)
- **Trajectory Diversity**: 0.65 (moderate variation)
- **Longevity Score**: 62.1% (Period 1 → Period 3)

---

## Data Requirements

### Google Drive Structure

Organize documents in Google Drive with consistent naming:

```
Root Folder/
├── Bristol-2021-ActionPlan.pdf
├── Bristol-2022-RAG.pdf
├── Oxford-2021-ActionPlan.docx
├── Oxford-2022-RAG.docx
└── ...
```

**Naming Convention:**
- `{Institution}-{Year}-{DocType}.{ext}`
- `{Year}-{Institution}-{DocType}.{ext}`
- DocType: `ActionPlan`, `RAG`, `AP1`, `RAG_AP1`, etc.

### Input CSV Format

For categorization scripts, CSV should have:
- **Action column**: Text of each action item
- Optionally: Institution, Document Name, etc.

---

## Performance Metrics

### Categorization Accuracy

Based on ground truth evaluation (n=341 actions):

| Metric | Average |
|--------|---------|
| Sensitivity (Recall) | 0.82 |
| Specificity | 0.95 |

See `evaluate_multi_col.py` output for per-category metrics.

### Processing Speed

| Task | Time (approx) |
|------|---------------|
| Extract 100 actions | 15 min |
| Categorize 3,410 actions | 3 sec |
| Full RAG workflow | 2-3 hours |
| Generate visualizations | 30 sec/institution |

**Note**: RAG workflow time depends on API quotas and dataset size.

---

## Advanced Usage

### Checkpoint Resume

If `rag_alignment_workflow_unpaired8.py` is interrupted:

```bash
python rag_alignment_workflow_unpaired8.py
# Prompts: "Resume from checkpoint? (y/n)"
```

Workflow resumes from last saved state.

### Custom Categories

To add/modify categories:

1. Edit `keyword_categories_4.json`
2. Add category with weighted keywords
3. Re-run `predefined_categories6.py`

**Example:**
```json
{
  "My New Category": {
    "specific keyword": 2.0,
    "related term": 1.5,
    "common phrase": 1.0
  }
}
```

### Batch Processing Multiple Institutions

The RAG workflow automatically processes all institutions in the input CSV. Results are institution-specific.

### Export Formats

Outputs are CSV (comma-separated values) for easy import into:
- Excel / Google Sheets
- R / Python (pandas)
- Statistical software (SPSS, Stata)
- Database systems

---

## Troubleshooting

### Common Issues

#### "Credentials file not found"
**Solution**: Download `credentials.json` from Google Cloud Console. See [INSTALL.md](INSTALL.md).

#### "NLTK data not found"
**Solution**:
```bash
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet stopwords
```

#### "Quota exceeded" (429 error)
**Solution**: Wait 1 minute for quota reset. Script has automatic rate limiting.

#### "Column 'Action' not found"
**Solution**: Check your CSV has the expected column name. Edit `ACTION_COLUMN_NAME` in script if needed.

### Debug Mode

For detailed logging, add to script:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Contributing

This is a research project accompanying a published paper. While primarily for reproducibility, improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

**Focus areas for contribution:**
- Additional document format support
- Performance optimizations
- Multilingual support
- Alternative categorization approaches

---

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@article{jackson2025technician,
  title={Synergies and Gaps in Technical Skills Development in UK Universities:
         A Semi-Quantitative Analysis of 'Technician Commitment' Action Plans
         and Progress Reports using Natural Language Processing},
  author={Jackson, Samuel J},
  year={2025},
  month={August}
}
```

**Code Repository:**
```
https://github.com/Sammjjj/TCDOC_EXTRACT
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Technician Commitment**: Data source for university action plans
- **Google Cloud Platform**: Gemini AI and Drive API
- **NLTK Project**: Natural language processing tools
- **scikit-learn**: Machine learning utilities
- **UK Research Community**: Feedback and validation

---

## Related Resources

- [Technician Commitment Website](https://www.technicians.org.uk/)
- [TALENT Commission Report](https://www.technicians.org.uk/talent)
- [Google Gemini Documentation](https://ai.google.dev/gemini-api/docs)
- [NLTK Documentation](https://www.nltk.org/)

---

## Contact

**Author**: Dr. Samuel J Jackson

For questions about:
- **Methodology**: See [Methods.txt](Methods.txt)
- **Installation**: See [INSTALL.md](INSTALL.md)
- **API Usage**: See [API_REFERENCE.md](API_REFERENCE.md)
- **Research**: Contact via GitHub issues

---

## Version History

- **v1.0** (August 2025): Initial public release
  - Complete analysis pipeline
  - 26-category taxonomy
  - RAG trajectory analysis
  - Multi-institution support
  - Checkpoint resumability

---

## Project Status

✅ **Stable** - Published research toolkit

This codebase is the finalized version accompanying the published paper. It is maintained for reproducibility and community use.

---

**Last Updated**: December 2025
