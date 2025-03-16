"""
Download random food and non-food images for testing the food tracker
Uses Unsplash API to fetch royalty-free images
"""

import requests
import logging
from pathlib import Path
import shutil
import random
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("image_downloader")

# Constants
TEST_IMAGES_DIR = Path("test_images")
TEST_IMAGES_DIR.mkdir(exist_ok=True)

# Unsplash API doesn't require auth for these sample URLs
UNSPLASH_COLLECTIONS = {
    "food": [
        "https://images.unsplash.com/photo-1546069901-ba9599a7e63c",  # Salad
        "https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445",  # Pancakes
        "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38",  # Pizza
        "https://images.unsplash.com/photo-1540189549336-e6e99c3679fe",  # Vegetable bowl
        "https://images.unsplash.com/photo-1529042410759-befb1204b468",  # Bento box
        "https://images.unsplash.com/photo-1512621776951-a57141f2eefd",  # Vegetable tray
        "https://images.unsplash.com/photo-1534939561126-855b8675edd7",  # Burger
        "https://images.unsplash.com/photo-1563729784474-d77dbb933a9e",  # Ramen
        "https://images.unsplash.com/photo-1563805042-7684c019e1cb",  # Sushi
        "https://images.unsplash.com/photo-1571091718767-18b5b1457add",  # Ice cream
    ],
    "non_food": [
        "https://images.unsplash.com/photo-1580757468214-c73f7062a5cb",  # Mountain
        "https://images.unsplash.com/photo-1579353977828-2a4eab540b9a",  # Sunset
        "https://images.unsplash.com/photo-1556139902-7367723b7e9e",  # Building
        "https://images.unsplash.com/photo-1502082553048-f009c37129b9",  # Trees
        "https://images.unsplash.com/photo-1494059980473-813e73ee784b",  # Beach
        "https://images.unsplash.com/photo-1457369804613-52c61a468e7d",  # Car
        "https://images.unsplash.com/photo-1504450874802-0ba2bcd9b5ae",  # Flowers
        "https://images.unsplash.com/photo-1484312152213-d713e8b7c053",  # Office desk
        "https://images.unsplash.com/photo-1534278931827-8a259344abe7",  # Dog
        "https://images.unsplash.com/photo-1530908295418-a12e326966ba",  # Book
    ]
}

def get_random_images(category, count):
    """Get random images from a category"""
    return random.sample(UNSPLASH_COLLECTIONS[category], min(count, len(UNSPLASH_COLLECTIONS[category])))

def download_image(url, save_path, category, index):
    """Download an image from a URL"""
    try:
        # Add random params to avoid caching
        full_url = f"{url}?random={random.randint(1, 1000)}&w=800&q=80"
        response = requests.get(full_url, stream=True)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)

        logger.info(f"✓ Downloaded {category} image {index+1} to {save_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        return False

def main(food_count=6, non_food_count=3):
    """Download food and non-food images"""
    logger.info(f"Downloading {food_count} food images and {non_food_count} non-food images")
    
    # Get random image URLs
    food_urls = get_random_images("food", food_count)
    non_food_urls = get_random_images("non_food", non_food_count)
    
    # Download food images
    for i, url in enumerate(food_urls):
        save_path = TEST_IMAGES_DIR / f"food_{i+1}.jpg"
        download_image(url, save_path, "food", i)
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Download non-food images
    for i, url in enumerate(non_food_urls):
        save_path = TEST_IMAGES_DIR / f"non_food_{i+1}.jpg"
        download_image(url, save_path, "non-food", i)
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    logger.info(f"Downloaded {food_count + non_food_count} images to {TEST_IMAGES_DIR}")
    logger.info("Ready to run food_tracker_poc.py for testing")

if __name__ == "__main__":
    main()
