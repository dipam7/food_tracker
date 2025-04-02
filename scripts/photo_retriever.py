# scripts/photo_retriever.py
"""
Selectively retrieves food photos from Google Photos
"""

import os
import json
import datetime
from pathlib import Path
import logging
from google_photos_auth import get_google_photos_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("photo_retriever")

# Constants
FOOD_KEYWORDS = ['food', 'meal', 'breakfast', 'lunch', 'dinner', 'snack', 'dish', 'plate']
PHOTOS_DIR = Path("retrieved_photos")
METADATA_FILE = PHOTOS_DIR / "metadata.json"
PHOTOS_DIR.mkdir(exist_ok=True)

def get_date_range(hours=None, days=None):
    """Get date range for the past hours or days"""
    end_date = datetime.datetime.now()
    
    if hours is not None:
        start_date = end_date - datetime.timedelta(hours=hours)
    elif days is not None:
        start_date = end_date - datetime.timedelta(days=days)
    else:
        # Default to last day if neither specified
        start_date = end_date - datetime.timedelta(days=1)
    
    # Format for Google Photos API
    return {
        'ranges': [{
            'startDate': {
                'year': start_date.year,
                'month': start_date.month,
                'day': start_date.day
            },
            'endDate': {
                'year': end_date.year,
                'month': end_date.month,
                'day': end_date.day
            }
        }]
    }

def search_food_photos(service, hours=None, days=None, max_results=50):
    """Search for food photos based on date and content categories"""
    try:
        # Prepare the search request with appropriate time range
        if hours is not None:
            logger.info(f"Searching for photos in the past {hours} hours...")
            date_filter = get_date_range(hours=hours)
        else:
            logger.info(f"Searching for photos in the past {days} days...")
            date_filter = get_date_range(days=days)
        
        # Use date filter
        search_request = {
            'pageSize': max_results,
            'filters': {
                'dateFilter': date_filter,
                # Content categories are deprecated, so we'll filter after retrieval
            }
        }
        
        # Execute the search
        results = service.mediaItems().search(body=search_request).execute()
        items = results.get('mediaItems', [])
        
        if not items:
            logger.info('No photos found for the given criteria.')
            return []
            
        logger.info(f'Found {len(items)} media items in the specified time range.')
        
        # Filter out videos and only keep photos
        photos_only = []
        for item in items:
            # Check if it's a photo by looking at mimeType or filename extension
            mime_type = item.get('mimeType', '').lower()
            filename = item.get('filename', '').lower()
            creation_time = item.get('mediaMetadata', {}).get('creationTime', '')
            
            # Skip videos - only keep images
            if 'video' in mime_type or filename.endswith(('.mp4', '.mov', '.avi', '.wmv', '.flv', '.mkv')):
                logger.info(f"Skipping video: {filename} ({creation_time})")
                continue
                
            logger.info(f"Found photo: {filename} ({creation_time})")
            photos_only.append(item)
        
        logger.info(f'Filtered to {len(photos_only)} photos (excluding videos).')
        
        # TEMPORARILY DISABLED KEYWORD FILTERING FOR TESTING
        # Just return the most recent photos for testing
        max_photos = min(5, len(photos_only))
        logger.info(f'Taking the {max_photos} most recent photos for testing')
        return photos_only[:max_photos]
        
    except Exception as e:
        logger.error(f"Error searching photos: {str(e)}")
        return []

def download_photos(service, photos, max_photos=5):
    """Download photos and save metadata"""
    if not photos:
        logger.info("No photos to download")
        return []
    
    logger.info(f"Downloading up to {max_photos} photos...")
    downloaded = []
    metadata = []
    
    for i, photo in enumerate(photos[:max_photos]):
        try:
            photo_id = photo['id']
            filename = photo['filename']
            creation_time = photo.get('mediaMetadata', {}).get('creationTime', '')
            
            # Get download URL (baseUrl needs to be appended with '=d' for download)
            download_url = photo['baseUrl'] + '=d'
            
            # Save metadata
            photo_metadata = {
                'id': photo_id,
                'filename': filename,
                'creation_time': creation_time,
                'local_path': str(PHOTOS_DIR / filename),
                'original_metadata': photo.get('mediaMetadata', {})
            }
            metadata.append(photo_metadata)
            
            # Download the photo
            import requests
            response = requests.get(download_url)
            if response.status_code == 200:
                with open(PHOTOS_DIR / filename, 'wb') as f:
                    f.write(response.content)
                downloaded.append(photo_metadata)
                logger.info(f"Downloaded {filename}")
            else:
                logger.error(f"Failed to download {filename}: HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error downloading photo {photo.get('filename', 'unknown')}: {str(e)}")
    
    # Save metadata to file
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Downloaded {len(downloaded)} photos. Metadata saved to {METADATA_FILE}")
    return downloaded

def main():
    """Main function to retrieve recent food photos"""
    logger.info("Getting Google Photos service...")
    service = get_google_photos_service()
    
    if not service:
        logger.error("Failed to create Google Photos service")
        return
        
    # Search for very recent photos (last 4 hours)
    recent_photos = search_food_photos(service, hours=4, max_results=50)
    
    # Download photos
    if recent_photos:
        logger.info(f"Found {len(recent_photos)} recent photos. Downloading for testing...")
        downloaded = download_photos(service, recent_photos)
        logger.info(f"Downloaded {len(downloaded)} photos to {PHOTOS_DIR}")
    else:
        logger.info("No photos found in the recent time period")

if __name__ == "__main__":
    main()
