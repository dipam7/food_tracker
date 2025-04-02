#\!/usr/bin/env python3
"""
Food Tracker - Main script (Asynchronous version)

Combines Google Photos authentication, photo retrieval, and food analysis
into a single workflow that can be scheduled to run regularly.
"""

import os
import json
import csv
import datetime
import shutil
import asyncio
import concurrent.futures
import base64
from pathlib import Path
import logging
from dotenv import load_dotenv
from tqdm import tqdm
import aiohttp

# Import functionality from other scripts
from google_photos_auth import get_google_photos_service
from food_tracker_poc import FoodAnalysis

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("food_tracker.log"), logging.StreamHandler()]
)
logger = logging.getLogger("food_tracker")

# Constants
PHOTOS_DIR = Path("retrieved_photos")
PHOTOS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("analysis_results")
RESULTS_DIR.mkdir(exist_ok=True)
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openai").lower()
MAX_WORKERS = 5  # Maximum number of concurrent workers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def get_date_range(days=7):
    """Get date range for the past week"""
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
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

def retrieve_photos(service, days=7, max_results=50):
    """Retrieve photos from Google Photos"""
    logger.info(f"Retrieving photos from the past {days} days...")
    
    try:
        # Prepare the search request
        date_filter = get_date_range(days)
        
        # Use date filter
        search_request = {
            'pageSize': max_results,
            'filters': {
                'dateFilter': date_filter,
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
            
            # Skip videos - only keep images
            if 'video' in mime_type or filename.endswith(('.mp4', '.mov', '.avi', '.wmv', '.flv', '.mkv')):
                logger.info(f"Skipping video: {filename}")
                continue
                
            photos_only.append(item)
        
        logger.info(f'Filtered to {len(photos_only)} photos (excluding videos).')
        return photos_only
        
    except Exception as e:
        logger.error(f"Error retrieving photos: {str(e)}")
        return []

async def download_single_photo(photo, session, pbar=None):
    """Download a single photo asynchronously"""
    try:
        photo_id = photo['id']
        filename = photo['filename']
        creation_time = photo.get('mediaMetadata', {}).get('creationTime', '')
        file_path = PHOTOS_DIR / filename
        
        # Skip downloading if already exists
        if file_path.exists():
            logger.info(f"Photo {filename} already exists, skipping download")
            return {
                'id': photo_id,
                'filename': filename,
                'creation_time': creation_time,
                'local_path': str(file_path),
                'original_metadata': photo.get('mediaMetadata', {}),
                'success': True
            }
        
        # Get download URL (baseUrl needs to be appended with '=d' for download)
        download_url = photo['baseUrl'] + '=d'
        
        # Use aiohttp for async HTTP requests
        async with session.get(download_url) as response:
            if response.status == 200:
                content = await response.read()
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                result = {
                    'id': photo_id,
                    'filename': filename,
                    'creation_time': creation_time,
                    'local_path': str(file_path),
                    'original_metadata': photo.get('mediaMetadata', {}),
                    'success': True
                }
                if pbar:
                    pbar.update(1)
                return result
            else:
                logger.error(f"Failed to download {filename}: HTTP {response.status}")
                if pbar:
                    pbar.update(1)
                return {'filename': filename, 'success': False, 'error': f"HTTP {response.status}"}
    
    except Exception as e:
        logger.error(f"Error downloading photo {photo.get('filename', 'unknown')}: {str(e)}")
        if pbar:
            pbar.update(1)
        return {'filename': photo.get('filename', 'unknown'), 'success': False, 'error': str(e)}

async def download_photos_async(photos, max_photos=20):
    """Download photos asynchronously"""
    if not photos:
        logger.info("No photos to download")
        return []
    
    photo_subset = photos[:max_photos]
    logger.info(f"Downloading up to {len(photo_subset)} photos asynchronously...")
    
    metadata = []
    downloaded = []
    
    # Create aiohttp session
    async with aiohttp.ClientSession() as session:
        # Create progress bar
        with tqdm(total=len(photo_subset), desc="Downloading photos") as pbar:
            # Create tasks for each photo
            tasks = [download_single_photo(photo, session, pbar) for photo in photo_subset]
            
            # Wait for all downloads to complete
            results = await asyncio.gather(*tasks)
            
            # Process results
            for result in results:
                if result.get('success', False):
                    metadata.append({
                        'id': result.get('id'),
                        'filename': result.get('filename'),
                        'creation_time': result.get('creation_time'),
                        'local_path': result.get('local_path'),
                        'original_metadata': result.get('original_metadata', {})
                    })
                    downloaded.append(result.get('local_path'))
    
    # Save metadata to file
    metadata_file = PHOTOS_DIR / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Downloaded {len(downloaded)} photos. Metadata saved to {metadata_file}")
    return downloaded

def copy_test_images():
    """Copy test images to retrieved_photos directory for testing"""
    test_images_dir = Path("test_images")
    if not test_images_dir.exists() or not any(test_images_dir.iterdir()):
        logger.info("No test images found to copy")
        return []
    
    copied = []
    for img_path in tqdm(list(test_images_dir.iterdir()), desc="Copying test images"):
        if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
            dest_path = PHOTOS_DIR / img_path.name
            if not dest_path.exists():  # Skip if already exists
                shutil.copy2(img_path, dest_path)
                copied.append(dest_path)
    
    logger.info(f"Copied {len(copied)} test images to {PHOTOS_DIR}")
    return copied

def encode_image_to_base64(image_path):
    """Encode image to base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        return None

async def encode_images_parallel(image_paths):
    """Encode multiple images to base64 in parallel"""
    logger.info(f"Encoding {len(image_paths)} images in parallel...")
    
    encoded_images = {}
    
    # Use ThreadPoolExecutor for CPU-bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(pool, encode_image_to_base64, path)
            for path in image_paths
        ]
        
        # Process results as they complete
        with tqdm(total=len(image_paths), desc="Encoding images") as pbar:
            for i, (path, future) in enumerate(zip(image_paths, asyncio.as_completed(futures))):
                try:
                    encoded = await future
                    if encoded:
                        encoded_images[path] = encoded
                except Exception as e:
                    logger.error(f"Error encoding {path}: {str(e)}")
                pbar.update(1)
    
    logger.info(f"Encoded {len(encoded_images)} images successfully")
    return encoded_images

async def analyze_with_openai_async(image_path, base64_image):
    """Analyze a single image with OpenAI asynchronously"""
    if not base64_image:
        return {"error": "Failed to encode image", "is_food": None}
    
    try:
        import instructor
        from openai import AsyncOpenAI
        
        # Use AsyncOpenAI client for async operations
        client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            response_model=FoodAnalysis,
            max_tokens=1000,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Analyze this image. If it's not food, set is_food to false. If it is food, identify the items and nutrition info."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"OpenAI analysis error for {image_path.name}: {str(e)}")
        return {"error": f"OpenAI analysis failed: {str(e)}", "is_food": None}

async def analyze_with_anthropic_async(image_path, base64_image):
    """Analyze a single image with Anthropic asynchronously"""
    if not base64_image:
        return {"error": "Failed to encode image", "is_food": None}
    
    try:
        import instructor
        from anthropic import AsyncAnthropic
        
        # Use AsyncAnthropic client for async operations
        client = instructor.patch(AsyncAnthropic(api_key=ANTHROPIC_API_KEY))
        
        response = await client.messages.create(
            model="claude-3-opus-20240229",
            response_model=FoodAnalysis,
            max_tokens=1000,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Analyze this image. If it's not food, set is_food to false. If it is food, identify the items and nutrition info."},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}}
                    ]
                }
            ]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Anthropic analysis error for {image_path.name}: {str(e)}")
        return {"error": f"Anthropic analysis failed: {str(e)}", "is_food": None}

async def analyze_image_async(image_path, encoded_image, provider=DEFAULT_PROVIDER):
    """Analyze a single image asynchronously"""
    filename = image_path.name
    logger.info(f"Analyzing {filename} with {provider}")
    
    if provider == "openai":
        result = await analyze_with_openai_async(image_path, encoded_image)
    else:
        result = await analyze_with_anthropic_async(image_path, encoded_image)
    
    base_result = {
        "filename": filename,
        "is_food": result.is_food if isinstance(result, FoodAnalysis) else None,
        "provider": provider,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    if isinstance(result, FoodAnalysis):
        base_result["analysis"] = result.model_dump()
    else:
        base_result["error"] = result.get("error", "Unknown error")
        
    return base_result

async def analyze_photos_async(provider=DEFAULT_PROVIDER, max_concurrent=3):
    """Analyze all photos in the retrieved_photos directory asynchronously"""
    if not PHOTOS_DIR.exists() or not any(PHOTOS_DIR.iterdir()):
        logger.error(f"No images found in {PHOTOS_DIR} directory")
        return []
    
    logger.info(f"Starting food image analysis using {provider}")
    
    # Find all images in the directory
    image_paths = []
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        image_paths.extend(PHOTOS_DIR.glob(f"*{ext}"))
    
    if not image_paths:
        logger.info(f"No images found in {PHOTOS_DIR}")
        return []
    
    logger.info(f"Found {len(image_paths)} images to analyze")
    
    # First encode all images in parallel
    encoded_images = await encode_images_parallel(image_paths)
    
    # Then analyze all encoded images with controlled concurrency
    results = []
    sem = asyncio.Semaphore(max_concurrent)  # Limit concurrent API calls
    
    async def analyze_with_semaphore(path):
        async with sem:  # This ensures we only run max_concurrent analyses at a time
            encoded = encoded_images.get(path)
            if not encoded:
                logger.error(f"No encoded data for {path}")
                return {
                    "filename": path.name,
                    "error": "Failed to encode image",
                    "is_food": None,
                    "provider": provider,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            return await analyze_image_async(path, encoded, provider)
    
    # Create tasks for each image
    tasks = [analyze_with_semaphore(path) for path in image_paths]
    
    # Process results as they complete
    with tqdm(total=len(tasks), desc="Analyzing images") as pbar:
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                results.append(result)
                
                status = "✓ Food" if result.get('is_food') else "✗ Not food"
                if result.get('is_food') is None:
                    status = "\! Error"
                logger.info(f"  {status}")
                
            except Exception as e:
                logger.error(f"Task error: {str(e)}")
            pbar.update(1)
    
    return results

def save_results_to_json(results):
    """Save analysis results to JSON file"""
    if not results:
        logger.info("No results to save")
        return None
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"results_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved analysis results to {result_file}")
    return result_file

def save_results_to_csv(results):
    """Save analysis results to CSV file"""
    if not results:
        logger.info("No results to save")
        return None
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = RESULTS_DIR / f"results_{timestamp}.csv"
    
    # Extract relevant fields for CSV
    csv_data = []
    for result in results:
        if result.get('is_food') and 'analysis' in result:
            analysis = result['analysis']
            nutrition = analysis.get('nutritional_info', {})
            
            row = {
                'filename': result['filename'],
                'date': result.get('timestamp', '').split('T')[0],
                'is_food': result.get('is_food', False),
                'meal_type': analysis.get('meal_type', ''),
                'food_items': ', '.join(analysis.get('food_items', [])),
                'calories': nutrition.get('calories', 0),
                'protein_g': nutrition.get('protein_g', 0),
                'carbs_g': nutrition.get('carbs_g', 0),
                'fat_g': nutrition.get('fat_g', 0),
                'fiber_g': nutrition.get('fiber_g', 0),
                'confidence': analysis.get('confidence', 0),
                'notes': analysis.get('analysis_notes', '')
            }
            csv_data.append(row)
        else:
            # For non-food or error results
            row = {
                'filename': result['filename'],
                'date': result.get('timestamp', '').split('T')[0],
                'is_food': result.get('is_food', False),
                'meal_type': '',
                'food_items': '',
                'calories': 0,
                'protein_g': 0,
                'carbs_g': 0,
                'fat_g': 0,
                'fiber_g': 0,
                'confidence': 0,
                'notes': result.get('error', '')
            }
            csv_data.append(row)
    
    # Write to CSV
    if csv_data:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
        
        logger.info(f"Saved CSV report to {csv_file}")
        return csv_file
    
    return None

async def main_async():
    """Asynchronous main function that runs the full food tracking pipeline"""
    logger.info("Starting Food Tracker (Async version)")
    
    # Step 1: Authenticate with Google Photos
    logger.info("Authenticating with Google Photos...")
    service = get_google_photos_service()
    
    if not service:
        logger.error("Failed to authenticate with Google Photos. Exiting.")
        return
    
    # Step 2: Retrieve photos from the past 7 days
    photos = retrieve_photos(service, days=7)
    
    if not photos:
        logger.info("No Google Photos found. Will continue with test images.")
    else:
        # Step 3: Download photos
        await download_photos_async(photos, max_photos=10)  # Limit to 10 photos for testing
    
    # Step 4: Copy test images (for testing/development)
    copy_test_images()
    
    # Step 5: Analyze all photos (with concurrency control)
    results = await analyze_photos_async(provider=DEFAULT_PROVIDER, max_concurrent=3)
    
    # Step 6: Save results
    save_results_to_json(results)
    csv_file = save_results_to_csv(results)
    
    # Step 7: Summarize results
    food_count = sum(1 for r in results if r.get("is_food"))
    error_count = sum(1 for r in results if r.get("is_food") is None)
    
    logger.info(f"\nAnalysis complete: {food_count} food images out of {len(results)} total")
    logger.info(f"Errors: {error_count}")
    
    if csv_file:
        logger.info(f"CSV report saved to {csv_file}")
    
    logger.info("Food Tracker complete\!")

# Define an entry point for running the async code
def main():
    """Entry point for the script"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
