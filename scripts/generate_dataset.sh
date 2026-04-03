#!/bin/bash
# Generate dataset by running scripts/run.py for each config file

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

# Config files to process (one per city)
CONFIGS=(
    # "config/boston_config_test.yaml"
    # "config/chicago_config_test.yaml"
    # "config/nyc_config_test.yaml"
    # "config/sf_config_test.yaml"
    "config/la_config_test.yaml"
)

echo "Starting dataset generation for ${#CONFIGS[@]} configs..."
echo "=========================================="

# Run scripts/run.py for each config
for config in "${CONFIGS[@]}"; do
    if [ ! -f "$config" ]; then
        echo "Warning: Config file not found: $config"
        continue
    fi
    
    echo ""
    echo "Processing: $config"
    echo "----------------------------------------"
    python scripts/run.py --config "$config"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: $config"
    else
        echo "✗ Failed: $config"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "Dataset generation complete for all configs!"
