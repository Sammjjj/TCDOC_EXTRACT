import os
import io
import csv
import re
import sys
import pdfplumber # <--- ADDED: Library to read PDFs
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# --- Vertex AI Setup ---
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.api_core.exceptions import NotFound

# --- CONFIGURATION ---
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'
LOCATION = 'us-central1'
OUTPUT_CSV_FILE = 'actions_output2.csv'

# Allow model and project to be set via env vars or runtime input. To use Gemini, set GCP_MODEL_ID to e.g. 'gemini-2.5-pro' or a specific version like 'gemini-2.5-pro-002'
PROJECT_ID = 'tcdocext'
MODEL_ID = os.environ.get('GCP_MODEL_ID', 'gemini-2.5-pro')

# OAuth2 Authentication
def authenticate():
    """Handles OAuth2 authentication and returns valid credentials."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"ERROR: Credentials file '{CREDENTIALS_FILE}' not found.")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    return creds

# List files in a Drive folder
def get_files_from_folder(service, folder_id):
    query = f"'{folder_id}' in parents and trashed=false"
    try:
        response = service.files().list(
            q=query,
            pageSize=100,
            fields='nextPageToken, files(id, name, mimeType)'
        ).execute()
        return response.get('files', [])
    except HttpError as error:
        print(f"An error occurred while listing files: {error}")
        return []

# Function to handle Google Docs and PDF files
def get_text_from_file(service, file_id, mime_type):
    """
    Extracts text from a Google Doc by exporting it or from a PDF by
    downloading it and using pdfplumber to read it.
    """
    print(f'  > Extracting text from {mime_type}...')
    try:
        # --- Handle Google Docs ---
        if 'google-apps.document' in mime_type:
            request = service.files().export_media(
                fileId=file_id,
                mimeType='text/plain'
            )
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return fh.getvalue().decode('utf-8')

        # --- Handle PDF Files ---
        elif 'application/pdf' in mime_type:
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                # Optional: print download progress
                # print(f"    - Downloading PDF: {int(status.progress() * 100)}%.")

            fh.seek(0)  # Go to the beginning of the downloaded file in memory
            full_text = ''
            with pdfplumber.open(fh) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:  # Ensure text was extracted from the page
                        full_text += text + '\n'
            return full_text

        # --- Handle other unsupported file types ---
        else:
            print(f"  > Unsupported file type: {mime_type}. Skipping.")
            return None

    except HttpError as error:
        print(f"  > An error occurred during file download/export: {error}")
        return None
    except Exception as e:
        # Catches other errors, including potential pdfplumber issues
        print(f"  > An error occurred processing the file: {e}")
        return None


# Re-flow text to remove artificial line breaks
def reflow_text(text: str) -> str:
    # Replace multiple newlines with single spaces (for within-paragraph breaks)
    text = re.sub(r'(?<=[^\.\?\!])\n+', ' ', text)
    # Replace two or more newlines with two newlines (paragraph breaks)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip() # Strip leading/trailing whitespace

def extract_actions_with_vertex_ai(text_content: str):
    if not text_content:
        return []
    if not PROJECT_ID or not MODEL_ID:
        print("ERROR: Both PROJECT_ID and MODEL_ID must be set.")
        sys.exit(1)

    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_ID)
    except Exception as e:
        print(f"ERROR initializing Vertex AI or loading model: {e}")
        return []

    prompt = (
        "Analyze the following document content and extract all specific action items, "
        "tasks, responsibilities, or planned activities. Present these as a clear, bulleted list. "
        "Each bullet point should represent a single, distinct action. "
        "Before listing actions, *first* briefly summarize the main topic of the document in one sentence. "
        "If the content is irrelevant or you find no actions, respond with ONLY 'No actions found'.\n\n"
        "Document Content:\n"
    )
    gen_config = GenerationConfig(temperature=0.0, max_output_tokens=4096)

    try:
        response = model.generate_content(
            [prompt + text_content],
            generation_config=gen_config,
            stream=False
        )
        raw = response.text

        if not raw or raw.strip().lower() == 'no actions found':
            return []

        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        
        actions = []
        for line in lines[1:]: # Start from the second line (skip summary)
            if line.startswith(('* ')) or line.startswith(('- ')) or line.startswith(('1. ')):
                actions.append(line)
        
        return actions

    except Exception as e:
        print(f"  > ERROR generating content: {e}")
        if hasattr(response, 'prompt_feedback'):
             print(f"  > Prompt Feedback: {response.prompt_feedback}")
        return []

# Main execution
def main():
    creds = authenticate()
    if not creds:
        return

    # Prompt for project/model if not set
    global PROJECT_ID, MODEL_ID
    if not PROJECT_ID:
        PROJECT_ID = input('Enter your GCP_PROJECT_ID: ').strip()
    if not MODEL_ID:
        MODEL_ID = input('Enter your GCP_MODEL_ID (e.g. gemini-1.5-pro-002): ').strip()

    try:
        service = build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"Could not build Google Drive service: {e}")
        return

    folder_id = '1Oweo-w7JWxTDiDLhcrAXaRflhQswF9_V' # HACK: Hardcoded for now.
    if not folder_id:
        print('Folder ID is required. Exiting.')
        return

    print(f"\nUsing model '{MODEL_ID}' in project '{PROJECT_ID}'...\n")
    files = get_files_from_folder(service, folder_id)
    print(f"Found {len(files)} files. Starting extraction and writing to {OUTPUT_CSV_FILE}...\n")

    actions_written_count = 0
    with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Document Name', 'Extracted Action']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in files:
            file_name = item.get('name', 'UNKNOWN_NAME')
            file_mime_type = item.get('mimeType', 'UNKNOWN_MIME')
            file_id = item.get('id', 'UNKNOWN_ID')

            print(f"Processing file: {file_name} ({file_mime_type})")
            text = get_text_from_file(service, file_id, file_mime_type)

            if isinstance(text, str) and text.strip():
                text = reflow_text(text)
                print("-" * 40)
                print(f"DEBUG: Text sent to AI for '{file_name}':")
                print(text[:500] + "..." if len(text) > 500 else text)
                print("-" * 40)
                actions = extract_actions_with_vertex_ai(text)
                
                if not actions:
                     print(f"  > No actions found or extracted for '{file_name}'.")

                for action_line in actions:
                    normalized_action = action_line.strip()
                    if normalized_action and normalized_action.lower() != 'no actions found' \
                        and (normalized_action.startswith(('* ')) or normalized_action.startswith(('- ')) or normalized_action.startswith(('1. '))):
                        
                        writer.writerow({'Document Name': file_name, 'Extracted Action': normalized_action})
                        actions_written_count += 1
                        print(f"  > Saved action to '{OUTPUT_CSV_FILE}'")

                    elif not normalized_action:
                        print(f"  > INFO: Skipped empty line from AI output for '{file_name}'.")
                    else:
                        print(f"  > INFO: Skipped non-bulleted/irrelevant line from AI output for '{file_name}': {normalized_action}")

            else:
                print(f"  > Skipping file '{file_name}' due to empty or unsupported content.")

    print(f"\nSuccess: Wrote a total of {actions_written_count} actions to '{OUTPUT_CSV_FILE}'.")

if __name__ == '__main__':
    main()