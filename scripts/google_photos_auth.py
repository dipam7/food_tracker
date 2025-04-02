# scripts/google_photos_auth.py
"""
Google Photos authentication and basic API testing
"""

import os
import pickle
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("google_photos")

# Constants
SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']
TOKEN_FILE = Path("token.pickle")
CREDENTIALS_FILE = Path("credentials.json")

def get_google_photos_service():
    """Authenticate and build the Google Photos service"""
    creds = None

    # Load existing token if available
    if TOKEN_FILE.exists():
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)

    # If credentials don't exist or are invalid, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                logger.error(f"Credentials file not found: {CREDENTIALS_FILE}")
                logger.error("Please download credentials.json from Google Cloud Console")
                return None
                
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for future runs
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)

    # Build and return the service
    try:
        service = build('photoslibrary', 'v1', credentials=creds, static_discovery=False)
        return service
    except Exception as e:
        logger.error(f"Error building Google Photos service: {str(e)}")
        return None

def list_recent_photos(service, max_results=10):
    """List recent photos to verify the API is working"""
    try:
        # Call the Photos v1 API
        results = service.mediaItems().list(pageSize=max_results).execute()
        items = results.get('mediaItems', [])

        if not items:
            logger.info('No photos found.')
            return []
            
        logger.info(f'Found {len(items)} photos:')
        for item in items:
            logger.info(f"{item['filename']} - {item.get('description', 'No description')}")
            
        return items
    except Exception as e:
        logger.error(f"Error listing photos: {str(e)}")
        return []

def main():
    """Main function to test Google Photos API connection"""
    logger.info("Authenticating with Google Photos API...")
    service = get_google_photos_service()
    
    if not service:
        logger.error("Failed to create Google Photos service")
        return
        
    logger.info("Authentication successful! Testing API by listing recent photos...")
    photos = list_recent_photos(service)
    
    logger.info(f"Successfully retrieved {len(photos)} photos from Google Photos")
    logger.info("Google Photos API integration test complete!")

if __name__ == "__main__":
    main()
