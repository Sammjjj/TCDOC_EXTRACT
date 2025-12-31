# Installation Guide

## Prerequisites

Before installing TCDOC_EXTRACT, ensure you have the following:

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 4GB (8GB recommended for large datasets)
- **Disk Space**: At least 2GB free space

### Google Cloud Platform Setup

This toolkit requires Google Cloud Platform access for:
1. **Google Drive API** - Document retrieval
2. **Vertex AI API** - Gemini language model access

#### Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Note your **Project ID** (you'll need this later)

#### Step 2: Enable Required APIs

Enable the following APIs in your project:
```bash
# Via Google Cloud Console:
# 1. Navigate to "APIs & Services" > "Library"
# 2. Search for and enable:
#    - Google Drive API
#    - Vertex AI API
```

Or use the `gcloud` CLI:
```bash
gcloud services enable drive.googleapis.com
gcloud services enable aiplatform.googleapis.com
```

#### Step 3: Create OAuth2 Credentials

1. Go to **APIs & Services** > **Credentials**
2. Click **Create Credentials** > **OAuth client ID**
3. Choose application type: **Desktop app**
4. Download the credentials JSON file
5. **Rename it to `credentials.json`** and place it in the project root directory

#### Step 4: Set Up Vertex AI

1. Navigate to **Vertex AI** in Google Cloud Console
2. Select your preferred region (default: `us-central1`)
3. Ensure billing is enabled (Vertex AI requires billing, but offers a free tier)

---

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Sammjjj/TCDOC_EXTRACT.git
cd TCDOC_EXTRACT
```

### 2. Create a Virtual Environment (Recommended)

#### On Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

The natural language processing modules require additional NLTK data:

```bash
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet stopwords
```

Or run this Python command:
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
```

### 5. Configure Google Cloud Credentials

#### Place credentials.json
Ensure your `credentials.json` file is in the project root directory:
```
TCDOC_EXTRACT/
├── credentials.json          ← Place here
├── document_extract2.py
├── requirements.txt
└── ...
```

#### Set Project ID
Update the `PROJECT_ID` variable in the following scripts:
- `document_extract2.py` (line 26)
- `rag_extract_simple12.py` (line 49)
- `rag_alignment_workflow_unpaired8.py` (line 44)

Change from:
```python
PROJECT_ID = 'tcdocext'  # or 'tcdocprog'
```

To your actual Project ID:
```python
PROJECT_ID = 'your-project-id-here'
```

### 6. First-Time Authentication

When you first run a script that accesses Google Drive, it will:
1. Open a browser window for OAuth2 authentication
2. Ask you to grant permissions
3. Save a `token.json` file for future use

**Important**: Keep `token.json` and `credentials.json` private and never commit them to version control.

---

## Verification

### Test Installation

Run this simple test to verify everything is installed correctly:

```bash
python -c "import pandas, nltk, sklearn, google.auth, vertexai, pdfplumber; print('✓ All dependencies installed successfully')"
```

Expected output:
```
✓ All dependencies installed successfully
```

### Test NLTK Data

```bash
python -c "import nltk; nltk.data.find('tokenizers/punkt'); print('✓ NLTK data downloaded')"
```

### Test Google Cloud Authentication

Run this to verify OAuth2 setup:

```python
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import os

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

if os.path.exists('credentials.json'):
    print('✓ credentials.json found')
    if os.path.exists('token.json'):
        print('✓ token.json found - you are authenticated')
    else:
        print('⚠ Run a script to complete first-time authentication')
else:
    print('✗ credentials.json NOT found - please download from Google Cloud Console')
```

---

## Troubleshooting

### Common Issues

#### ImportError: No module named 'X'
**Solution**: Ensure virtual environment is activated and run:
```bash
pip install -r requirements.txt
```

#### NLTK Data Not Found
**Error**: `LookupError: Resource punkt not found`

**Solution**:
```bash
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet stopwords
```

#### Credentials File Not Found
**Error**: `ERROR: Credentials file 'credentials.json' not found.`

**Solution**:
1. Download OAuth2 credentials from Google Cloud Console
2. Rename to `credentials.json`
3. Place in project root directory

#### Vertex AI Authentication Failed
**Error**: `google.auth.exceptions.DefaultCredentialsError`

**Solution**:
1. Ensure Vertex AI API is enabled in your project
2. Verify billing is enabled
3. Check `PROJECT_ID` matches your actual project

#### Permission Denied on Google Drive
**Error**: `HttpError 403: Forbidden`

**Solution**:
1. Delete `token.json`
2. Re-run script to re-authenticate
3. Ensure you grant all requested permissions

#### Rate Limit Exceeded
**Error**: `google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded`

**Solution**:
- Wait for quota to reset (usually 1 minute)
- Script automatically handles rate limiting in `rag_alignment_workflow_unpaired8.py`
- For persistent issues, check quota limits in Google Cloud Console

---

## Optional Configuration

### Increase API Rate Limits

If you have a paid Google Cloud account, you can increase rate limits:

1. Go to **APIs & Services** > **Quotas**
2. Search for "Vertex AI API"
3. Request quota increase

### Use Different Gemini Model

To use a different Gemini model version, set environment variable:

```bash
export GCP_MODEL_ID='gemini-2.0-flash-exp-001'
python document_extract2.py
```

Or edit the script directly:
```python
MODEL_ID = os.environ.get('GCP_MODEL_ID', 'your-preferred-model')
```

### Configure DOCX Support

For Microsoft Word document support:

```bash
pip install python-docx
```

This is optional - PDF and Google Docs work without it.

---

## Upgrading

To upgrade to the latest version:

```bash
cd TCDOC_EXTRACT
git pull origin main
pip install -r requirements.txt --upgrade
```

---

## Uninstallation

To completely remove the toolkit:

```bash
# Deactivate virtual environment
deactivate

# Remove directory
cd ..
rm -rf TCDOC_EXTRACT
```

**Note**: This will not remove your Google Cloud project or credentials. To fully clean up:
1. Delete OAuth2 credentials from Google Cloud Console
2. Disable APIs if no longer needed
3. Delete the Google Cloud project if created specifically for this

---

## Environment Variables Reference

| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `GCP_MODEL_ID` | Gemini model version | `gemini-2.5-pro` or `gemini-2.0-flash` | No |
| `GOOGLE_APPLICATION_CREDENTIALS` | Service account key (alternative auth) | None | No |

---

## Next Steps

After installation:
1. Review [README.md](README.md) for usage examples
2. Check [API_REFERENCE.md](API_REFERENCE.md) for detailed documentation
3. Read [Methods.txt](Methods.txt) for methodological background
4. Prepare your input data (see README.md)

---

## Support

If you encounter issues not covered here:
1. Check the [API Reference](API_REFERENCE.md) for configuration details
2. Review error messages carefully - they often indicate the exact issue
3. Verify all file paths and column names match expected formats
4. Ensure Google Cloud APIs are enabled and billing is active

---

## License

See [LICENSE](LICENSE) file for details.

## Citation

If you use this toolkit, please cite:

```
Jackson, S.J. (2025). Synergies and Gaps in Technical Skills Development
in UK Universities: A Semi-Quantitative Analysis of 'Technician Commitment'
Action Plans and Progress Reports using Natural Language Processing.
```

---

Last updated: December 2025
