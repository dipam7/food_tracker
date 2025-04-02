#\!/bin/bash
# Food Tracker Weekly Runner
# This script runs the food tracker every Sunday night to collect and analyze weekly food images

# Set up logging
LOG_DIR="$HOME/Desktop/Desktop - Dipam's MacBook Air/coding/food_tracker/logs"
LOG_FILE="$LOG_DIR/food_tracker_$(date +\%Y\%m\%d).log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Navigate to project directory
cd "$HOME/Desktop/Desktop - Dipam's MacBook Air/coding/food_tracker" || {
    echo "Failed to navigate to project directory" >> "$LOG_FILE"
    exit 1
}

# Activate Poetry environment and run the script
echo "=== Starting Food Tracker at $(date) ===" >> "$LOG_FILE" 2>&1
poetry run python scripts/food_tracker.py >> "$LOG_FILE" 2>&1
echo "=== Completed Food Tracker at $(date) ===" >> "$LOG_FILE" 2>&1

# Optional: Send a notification when done
osascript -e 'display notification "Weekly food tracking complete. Check logs for details." with title "Food Tracker"'
