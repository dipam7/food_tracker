import base64, json, logging, instructor
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from pydantic import BaseModel, Field
import os  # Still needed for os.getenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("food_tracker.log"), logging.StreamHandler()]
)
logger = logging.getLogger("food_tracker")

TEST_IMAGES_DIR = Path("test_images")
RESULTS_DIR = Path("test_results")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openai").lower()

TEST_IMAGES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

class NutritionalInfo(BaseModel):
    calories: float = Field(..., description="Approximate calories")
    protein_g: float = Field(..., description="Protein in grams")
    carbs_g: float = Field(..., description="Carbs in grams")
    fat_g: float = Field(..., description="Fat in grams")
    fiber_g: float = None
    vitamins_minerals: list = Field(..., description="Notable nutrients")

class FoodAnalysis(BaseModel):
    is_food: bool = Field(..., description="Is this food or not")
    food_items: list = None
    meal_type: str = None
    nutritional_info: NutritionalInfo = None
    confidence: float = Field(..., description="Confidence level (0-1)")
    analysis_notes: str = None

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        return None

def analyze_with_openai(image_path):
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return {"error": "Failed to encode image"}
    
    try:
        client = instructor.patch(OpenAI(api_key=OPENAI_API_KEY))
        
        response = client.chat.completions.create(
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
        logger.error(f"OpenAI analysis error: {str(e)}")
        return {"error": f"OpenAI analysis failed: {str(e)}"}

def analyze_with_anthropic(image_path):
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return {"error": "Failed to encode image"}
    
    try:
        client = instructor.patch(Anthropic(api_key=ANTHROPIC_API_KEY))
        
        response = client.messages.create(
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
        logger.error(f"Anthropic analysis error: {str(e)}")
        return {"error": f"Anthropic analysis failed: {str(e)}"}

def analyze_image(image_path, provider=DEFAULT_PROVIDER):
    filename = image_path.name
    logger.info(f"Analyzing {filename} with {provider}")
    
    if provider == "openai":
        result = analyze_with_openai(image_path)
    else:
        result = analyze_with_anthropic(image_path)
    
    base_result = {
        "filename": filename,
        "is_food": result.is_food if isinstance(result, FoodAnalysis) else None,
        "provider": provider,
        "timestamp": datetime.now().isoformat()
    }
    
    if isinstance(result, FoodAnalysis):
        base_result["analysis"] = result.model_dump()
    else:
        base_result["error"] = result.get("error", "Unknown error")
        
    return base_result

def main():
    if not TEST_IMAGES_DIR.is_dir() or not any(TEST_IMAGES_DIR.iterdir()):
        logger.error(f"No images found in {TEST_IMAGES_DIR} directory")
        return
    
    logger.info(f"Starting food image analysis using {DEFAULT_PROVIDER}")
    results = []
    
    for image_path in TEST_IMAGES_DIR.iterdir():
        if image_path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'):
            result = analyze_image(image_path)
            results.append(result)
            
            status = "✓ Food" if result.get('is_food') else "✗ Not food"
            if result.get('is_food') is None:
                status = "! Error"
            logger.info(f"  {status}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"results_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    food_count = sum(1 for r in results if r.get("is_food"))
    error_count = sum(1 for r in results if r.get("is_food") is None)
    
    logger.info(f"\nAnalysis complete: {food_count} food images out of {len(results)} total")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Results saved to {result_file}")

if __name__ == "__main__":
    main()