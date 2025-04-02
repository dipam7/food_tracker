# Food Tracker

A system that automatically analyzes your food photos from Google Photos and provides nutritional summaries.

## Overview

Food Tracker connects to your Google Photos account, retrieves images that contain food, and uses AI vision models to analyze nutritional content. It runs as a scheduled task to give you weekly summaries of your eating habits.

## Features

- **Google Photos Integration**: Automatically retrieves your food photos
- **AI-Powered Analysis**: Uses GPT-4o or Claude to identify food items and estimate nutritional content
- **Scheduled Reports**: Runs as a weekly cron job to track your eating patterns
- **Nutritional Insights**: Provides calories, macronutrients, and vitamin/mineral information
- **Multi-Model Support**: Works with both OpenAI and Anthropic vision models

## Requirements

- Python 3.10+
- Poetry for dependency management
- Google Photos API credentials
- OpenAI API key and/or Anthropic API key

## Setup

1. Clone this repository
2. Install dependencies: `poetry install`
3. Set up Google Photos API:
   - Create a project in Google Cloud Console
   - Enable the Google Photos Library API
   - Create OAuth credentials and download the JSON file
   - Place the credentials in the project directory
4. Configure environment variables in `.env`:
   ```
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   DEFAULT_PROVIDER=openai  # or anthropic
   GOOGLE_CREDENTIALS_FILE=path_to_credentials.json
   ```

## Usage

### Set up Google Photos Authentication
```
poetry run python scripts/google_photos_auth.py
```
This will:
1. Open a browser window to authenticate with Google Photos
2. Create a token.pickle file to store your credentials
3. Test the connection by listing your recent photos

### Download test images (development only)
```
poetry run python scripts/download-test-images.py
```

### Run the analyzer manually
```
poetry run python scripts/food_tracker_poc.py
```

### Configure as cron job
Add to your crontab to run weekly:
```
0 9 * * 1 cd /path/to/food_tracker && poetry run python scripts/food_tracker_poc.py
```

## Project Structure

- `scripts/food_tracker_poc.py`: Main analysis script
- `scripts/download-test-images.py`: Utility to download test images
- `scripts/google_photos_auth.py`: Google Photos API authentication script
- `credentials.json`: Google Cloud OAuth credentials
- `token.pickle`: Stored Google Photos authentication token
- `test_images/`: Directory of sample food and non-food images
- `test_results/`: Output JSON files with analysis results

## Development

1. Format code: `poetry run black .`
2. Run tests: `poetry run pytest`

## Next Steps

- [x] Implement Google Photos API authentication
- [ ] Build food photo filtering and retrieval from Google Photos
- [ ] Build weekly summary report generation
- [ ] Add visualization of eating patterns over time
- [ ] Create user dashboard for viewing results
