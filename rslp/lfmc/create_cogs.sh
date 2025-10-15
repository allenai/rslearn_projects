#!/bin/bash

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <gs://bucket/path>"
    exit 1
fi

GCS_PATH="$1"
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "Finding .tif files in $GCS_PATH..."

gsutil ls -r "${GCS_PATH}/**/*.tif" | grep -v '\.cog\.tif$' | while read -r GCS_FILE; do
    FILENAME=$(basename "$GCS_FILE")
    COG_FILE="${FILENAME%.tif}.cog.tif"
    GCS_DIR=$(dirname "$GCS_FILE")
    
    echo "Processing $FILENAME..."
    
    gsutil cp "$GCS_FILE" "$TEMP_DIR/$FILENAME"
    
    rio cogeo create "$TEMP_DIR/$FILENAME" "$TEMP_DIR/$COG_FILE" \
        --cog-profile deflate \
        --overview-level 5 \
        --quiet
    
    gsutil cp "$TEMP_DIR/$COG_FILE" "$GCS_DIR/$COG_FILE"
    
    rm "$TEMP_DIR/$FILENAME" "$TEMP_DIR/$COG_FILE"
    
    echo "Done with $COG_FILE"
done

echo "Done!"
