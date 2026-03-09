#!/bin/bash
# Aggregate LoS statistics over all channel files for each city.

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

# Cities to process (must match scene / dataset directory names)
CITIES=(
    "boston_1"
    "chicago_1"
    "nyc_1"
    "sf_1"
)

echo "Starting LoS aggregation for ${#CITIES[@]} cities..."
echo "=========================================="

for city in "${CITIES[@]}"; do
    echo ""
    echo "Processing city: $city"
    echo "----------------------------------------"
    python scripts/aggregate_los_stats.py --city "$city"
done

echo ""
echo "=========================================="
echo "LoS aggregation complete for all cities!"

