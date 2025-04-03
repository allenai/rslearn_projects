#!/bin/bash

# Script that looks at the files changed in the PR and figures out which test directories should be executed.

# Validate input argument
if [[ "$#" -ne 1 || ( "$1" != "unit" && "$1" != "integration" ) ]]; then
    echo "Usage: $0 <unit|integration>"
    exit 1
fi

TEST_TYPE=$1
# We **always** run the tests in tests/unit/*.py or tests/integration/*.py
TEST_DIRS=("tests/$TEST_TYPE/*.py")

# Get the list of changed files in the PR
CHANGED_FILES=$(git diff --name-only origin/master...HEAD)

# Find directories inside tests/$TEST_TYPE
for dir in tests/$TEST_TYPE/*/; do
    dir_name=$(basename "$dir")
    if echo "$CHANGED_FILES" | grep -q "^rslp/$dir_name/"; then
        TEST_DIRS+=("tests/$TEST_TYPE/$dir_name/**")
    fi
done

# Output the list of test directories
echo "${TEST_DIRS[@]}"
