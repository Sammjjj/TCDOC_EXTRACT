# Changelog

All notable changes to the TCDOC_EXTRACT project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-08-01

### Added
- Initial public release of TCDOC_EXTRACT toolkit
- Core extraction pipeline (`document_extract2.py`)
  - Google Drive API integration for document retrieval
  - PDF and Google Docs text extraction
  - Gemini AI-powered action item extraction
- Categorization system (`predefined_categories6.py`)
  - 26 predefined thematic categories
  - Weighted keyword lexicon with 500+ domain terms
  - NLTK-based text preprocessing with lemmatization
  - One-hot encoded output format
- Performance evaluation (`evaluate_multi_col.py`)
  - Confusion matrix-based metrics
  - Sensitivity and specificity calculations
  - Ground truth comparison
- Sub-category analysis
  - TF-IDF keyword identification (`subclus_calc_weight.py`)
  - Rule-based sub-category classification (`rule_based_class.py`)
  - 17 main categories with 3 sub-categories each
- RAG assessment analysis (`rag_extract_simple12.py`)
  - Structured table extraction with intelligent column detection
  - Multi-row action merging
  - Unstructured text extraction via Gemini AI
  - RAG status indicator parsing (Red/Amber/Green)
- Longitudinal tracking workflow (`rag_alignment_workflow_unpaired8.py`)
  - Within-period TF-IDF matching (AP ↔ RAG_AP)
  - Cross-period Gemini AI semantic matching (RAG_AP → AP)
  - Three-tier matching strategy
  - Trajectory classification (Continued, Related, Stopped, New)
  - 7 derived institutional metrics
  - Checkpoint-based resumability
  - Automatic rate limiting (15 req/min)
  - Batch processing optimization (10x cost reduction)
- Visualization tools
  - Multi-institution graphical summaries (`create_graphical_summary_multi_institution.py`)
  - Comprehensive text-based tables (`create_comprehensive_summary_multi_institution.py`)
  - Publication-quality PNG outputs
- Resource enumeration (`enumerate_resources.py`)
  - Whole-word phrase matching
  - Context display with terminal highlighting
  - External resource tracking
- Configuration files
  - `keyword_categories_4.json` - 26-category taxonomy
  - OAuth2 authentication setup

### Documentation
- Comprehensive README.md with quick start guide
- Detailed INSTALL.md with Google Cloud setup
- Complete API_REFERENCE.md with all module documentation
- Methods.txt with full methodological background
- CONTRIBUTING.md with contribution guidelines
- requirements.txt with all dependencies
- Apache 2.0 LICENSE

### Research
- Accompanies paper: "Synergies and Gaps in Technical Skills Development in UK Universities"
- Author: Dr. Samuel J Jackson
- Dataset: 3,410 action items from UK university Technician Commitment documents
- Validation: n=341 manually classified ground truth samples
- Performance: 82% sensitivity, 95% specificity (average across categories)

## [Unreleased]

### Planned
- Additional document format support (.odt, HTML)
- Multilingual support for non-English documents
- Enhanced visualization options (interactive plots)
- API wrapper for easier integration
- Docker containerization
- Web interface for non-technical users

---

## Version Numbering

This project uses [Semantic Versioning](https://semver.org/):
- **MAJOR** version: Incompatible API changes
- **MINOR** version: New functionality (backward-compatible)
- **PATCH** version: Bug fixes (backward-compatible)

## Release Types

- **[Major Release]**: Significant new features or breaking changes
- **[Minor Release]**: New features, backward-compatible
- **[Patch]**: Bug fixes and minor improvements
- **[Unreleased]**: Planned features for future versions

---

**Note**: This is a research toolkit. Version 1.0.0 represents the state of the codebase accompanying the published research paper. Future versions will maintain reproducibility of original results while adding new capabilities.

---

Last updated: December 2025
