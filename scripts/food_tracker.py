#\!/usr/bin/env python3
"""
Food Tracker - Cached Version with Improved Schema Validation

Uses proper schema validation with Instructor and disk-based caching to minimize API calls.
"""

import os
import json
import csv
import datetime
import shutil
import asyncio
import concurrent.futures
import base64
import hashlib
from pathlib import Path
import logging
from dotenv import load_dotenv
from tqdm import tqdm
import aiohttp
from pydantic import BaseModel, Field

# Import functionality from other scripts
from google_photos_auth import get_google_photos_service

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
CACHE_DIR = Path("api_cache")
CACHE_DIR.mkdir(exist_ok=True)
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openai").lower()
MAX_WORKERS = 5  # Maximum number of concurrent workers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MAX_RETRIES = 1  # Maximum number of retries for failed analyses

# Define Pydantic models for structured extraction
class FoodClassification(BaseModel):
    """Classification of whether an image contains food"""
    is_food: bool = Field(..., description="Whether the image contains food")

class NutritionalInfo(BaseModel):
    """Basic nutritional information for a food item"""
    calories: int = Field(..., description="Approximate calories in the meal")
    protein_g: int = Field(..., description="Protein content in grams")
    carbs_g: int = Field(..., description="Carbohydrate content in grams")
    fat_g: int = Field(..., description="Fat content in grams")

class FoodDetails(BaseModel):
    """Detailed information about food in an image"""
    meal_type: str = Field(..., description="Type of meal (breakfast, lunch, dinner, snack)")
    food_items: list[str] = Field(..., description="List of individual food items in the image")
    nutritional_info: NutritionalInfo = Field(..., description="Nutritional information for the meal")
    notes: str = Field(default="", description="Additional notes about the food")

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

def get_image_hash(base64_image):
    """Generate a hash for an image to use as cache key"""
    return hashlib.md5(base64_image.encode()).hexdigest()

def get_cached_result(image_hash, operation_type):
    """Get a cached result for an image hash if it exists"""
    cache_file = CACHE_DIR / f"{image_hash}_{operation_type}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading cache file {cache_file}: {str(e)}")
    return None

def save_to_cache(image_hash, operation_type, result):
    """Save a result to the cache"""
    cache_file = CACHE_DIR / f"{image_hash}_{operation_type}.json"
    try:
        with open(cache_file, 'w') as f:
            json.dump(result, f)
        return True
    except Exception as e:
        logger.error(f"Error writing cache file {cache_file}: {str(e)}")
        return False

async def classify_image_openai(image_path, base64_image):
    """Classify a single image as food or non-food with OpenAI"""
    if not base64_image:
        return {"error": "Failed to encode image", "is_food": None, "filename": image_path.name}
    
    # Generate hash for the image
    image_hash = get_image_hash(base64_image)
    
    # Check cache
    cache_result = get_cached_result(image_hash, "classify")
    if cache_result is not None:
        logger.info(f"Using cached classification result for image {image_hash[:8]}...")
        # Add filename to the cached result
        cache_result["filename"] = image_path.name
        return cache_result
    
    try:
        import instructor
        from openai import AsyncOpenAI
        
        # Use AsyncOpenAI client for async operations with instructor patching
        client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            response_model=FoodClassification,  # Use Pydantic model for structured output
            max_tokens=150,  # Reduced tokens since we only need classification
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Is this image of food? Answer yes or no."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        
        result = {
            "filename": image_path.name,
            "is_food": response.is_food,
            "confidence": 0.9,  # Placeholder confidence
            "provider": "openai",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Cache the result (without the filename to ensure consistent keys)
        cache_key = result.copy()
        del cache_key["filename"]
        save_to_cache(image_hash, "classify", cache_key)
        
        return result
        
    except Exception as e:
        logger.error(f"OpenAI classification error for {image_path.name}: {str(e)}")
        return {
            "filename": image_path.name,
            "error": f"OpenAI classification failed: {str(e)}", 
            "is_food": None,
            "provider": "openai",
            "timestamp": datetime.datetime.now().isoformat()
        }

async def analyze_food_openai(image_path, base64_image, retry_count=0):
    """Analyze food details for an image already known to contain food"""
    if not base64_image:
        return {"error": "Failed to encode image", "filename": image_path.name}
    
    # Generate hash for the image
    image_hash = get_image_hash(base64_image)
    
    # Check cache
    cache_result = get_cached_result(image_hash, "analyze")
    if cache_result is not None:
        logger.info(f"Using cached food analysis result for image {image_hash[:8]}...")
        # Add filename to the cached result
        cache_result["filename"] = image_path.name
        return cache_result
    
    try:
        import instructor
        from openai import AsyncOpenAI
        
        # Use AsyncOpenAI client for async operations with instructor patching
        client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            response_model=FoodDetails,  # Use Pydantic model for structured output
            max_tokens=1000,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "This image contains food. Analyze it to identify the food items, meal type, and nutritional information."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        
        # With instructor, we get a properly structured response matching our FoodDetails model
        result = {
            "filename": image_path.name,
            "is_food": True,
            "meal_type": response.meal_type,
            "food_items": response.food_items,
            "nutritional_info": response.nutritional_info.model_dump(),  # Use model_dump instead of dict
            "notes": response.notes,
            "provider": "openai",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Cache the result (without the filename to ensure consistent keys)
        cache_key = result.copy()
        del cache_key["filename"]
        save_to_cache(image_hash, "analyze", cache_key)
        
        return result
        
    except Exception as e:
        logger.error(f"OpenAI food analysis error for {image_path.name}: {str(e)}")
        
        # Retry logic
        if retry_count < MAX_RETRIES:
            logger.info(f"Retrying analysis for {image_path.name} (attempt {retry_count + 1}/{MAX_RETRIES})")
            await asyncio.sleep(1)  # Short delay before retry
            return await analyze_food_openai(image_path, base64_image, retry_count + 1)
        
        return {
            "filename": image_path.name,
            "is_food": True,  # We know it's food, just the detailed analysis failed
            "error": f"OpenAI analysis failed after {MAX_RETRIES + 1} attempts: {str(e)}",
            "provider": "openai",
            "timestamp": datetime.datetime.now().isoformat()
        }

async def classify_photos_async(image_paths, encoded_images, provider=DEFAULT_PROVIDER, max_concurrent=5):
    """Classify photos as food or non-food in parallel"""
    logger.info(f"Classifying {len(image_paths)} images as food or non-food...")
    
    results = []
    sem = asyncio.Semaphore(max_concurrent)  # Limit concurrent API calls
    
    async def classify_with_semaphore(path):
        async with sem:
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
            if provider == "openai":
                return await classify_image_openai(path, encoded)
            else:
                # Add support for Anthropic if needed
                return await classify_image_openai(path, encoded)  # Fallback to OpenAI for now
    
    # Create tasks for each image
    tasks = [classify_with_semaphore(path) for path in image_paths]
    
    # Process results as they complete
    food_images = []
    non_food_images = []
    error_images = []
    
    with tqdm(total=len(tasks), desc="Classifying images") as pbar:
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                results.append(result)
                
                # Categorize results
                if result.get('is_food') is True:
                    food_images.append((Path(result['filename']), encoded_images.get(next(p for p in image_paths if p.name == result['filename']))))
                    status = "✓ Food"
                elif result.get('is_food') is False:
                    non_food_images.append(result['filename'])
                    status = "✗ Not food"
                else:
                    error_images.append(result['filename'])
                    status = "\! Error"
                    
                logger.info(f"  {status}: {result['filename']}")
                
            except Exception as e:
                logger.error(f"Classification task error: {str(e)}")
            pbar.update(1)
    
    logger.info(f"Classification complete: {len(food_images)} food, {len(non_food_images)} non-food, {len(error_images)} errors")
    return results, food_images

async def analyze_food_photos_async(food_images, provider=DEFAULT_PROVIDER, max_concurrent=3):
    """Analyze only the food photos for nutritional content"""
    if not food_images:
        logger.info("No food images to analyze")
        return []
    
    logger.info(f"Analyzing {len(food_images)} food images for nutritional content...")
    
    results = []
    sem = asyncio.Semaphore(max_concurrent)  # Limit concurrent API calls
    
    async def analyze_with_semaphore(path_tuple):
        path, encoded = path_tuple
        async with sem:
            if not encoded:
                logger.error(f"No encoded data for {path}")
                return {
                    "filename": path.name,
                    "error": "Failed to encode image",
                    "is_food": True,  # We assume it's food since it passed classification
                    "provider": provider,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            if provider == "openai":
                return await analyze_food_openai(path, encoded)
            else:
                # Add support for Anthropic if needed
                return await analyze_food_openai(path, encoded)  # Fallback to OpenAI for now
    
    # Create tasks for each food image
    tasks = [analyze_with_semaphore(image) for image in food_images]
    
    # Process results as they complete
    with tqdm(total=len(tasks), desc="Analyzing food content") as pbar:
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                results.append(result)
                
                if "error" in result:
                    status = f"\! Analysis Error: {result['filename']}"
                else:
                    food_items = result.get('food_items', [])
                    if food_items:
                        status = f"✓ {result['filename']}: {', '.join(food_items[:3])}"
                        if len(food_items) > 3:
                            status += "..."
                    else:
                        status = f"✓ {result['filename']}: Analysis complete"
                logger.info(f"  {status}")
                
            except Exception as e:
                logger.error(f"Food analysis task error: {str(e)}")
            pbar.update(1)
    
    return results

def save_results_to_json(classification_results, analysis_results):
    """Save all results to JSON file"""
    all_results = classification_results.copy()
    
    # Update with detailed analysis for food items
    for analysis in analysis_results:
        for i, result in enumerate(all_results):
            if result['filename'] == analysis['filename']:
                all_results[i].update(analysis)
                break
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"results_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Saved analysis results to {result_file}")
    return result_file, all_results

def save_results_to_csv(combined_results):
    """Save analysis results to CSV file"""
    if not combined_results:
        logger.info("No results to save")
        return None
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = RESULTS_DIR / f"results_{timestamp}.csv"
    
    # Extract relevant fields for CSV with meal_description instead of notes/is_food
    csv_data = []
    for result in combined_results:
        row = {
            'filename': result['filename'],
            'date': result.get('timestamp', '').split('T')[0],
            'is_food': result.get('is_food', False),
            'meal_type': result.get('meal_type', ''),
            'meal_description': ', '.join(result.get('food_items', [])),
            'calories': result.get('nutritional_info', {}).get('calories', 0),
            'protein_g': result.get('nutritional_info', {}).get('protein_g', 0),
            'carbs_g': result.get('nutritional_info', {}).get('carbs_g', 0),
            'fat_g': result.get('nutritional_info', {}).get('fat_g', 0)
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
    logger.info("Starting Food Tracker (Cached Version)")
    
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
    
    # Step 5: Find all images in the directory
    image_paths = []
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        image_paths.extend(PHOTOS_DIR.glob(f"*{ext}"))
    
    if not image_paths:
        logger.info(f"No images found in {PHOTOS_DIR}")
        return
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Step 6: Encode all images in parallel
    encoded_images = await encode_images_parallel(image_paths)
    
    # Step 7: First stage - Classify all images as food/non-food
    classification_results, food_images = await classify_photos_async(
        image_paths, 
        encoded_images, 
        provider=DEFAULT_PROVIDER, 
        max_concurrent=5
    )
    
    # Step 8: Second stage - Only analyze food images for nutritional content
    if food_images:
        analysis_results = await analyze_food_photos_async(
            food_images, 
            provider=DEFAULT_PROVIDER, 
            max_concurrent=3
        )
    else:
        analysis_results = []
        logger.info("No food images found to analyze")
    
    # Step 9: Save results
    result_file, combined_results = save_results_to_json(classification_results, analysis_results)
    csv_file = save_results_to_csv(combined_results)
    
    # Step 10: Summarize results
    food_count = sum(1 for r in combined_results if r.get("is_food"))
    error_count = sum(1 for r in combined_results if r.get("is_food") is None)
    analyzed_count = len(analysis_results)
    
    logger.info(f"\nAnalysis complete:")
    logger.info(f"- {food_count} food images identified out of {len(image_paths)} total")
    logger.info(f"- {analyzed_count} food images analyzed with nutritional details")
    logger.info(f"- {error_count} errors in classification")
    
    if csv_file:
        logger.info(f"CSV report saved to {csv_file}")
    
    logger.info("Food Tracker complete\!")

# Define an entry point for running the async code
def main():
    """Entry point for the script"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
