#!/bin/bash

cd "statistics" || { echo "Directory not found: statistics"; exit 1; }
for file in *runs*; do
    echo -e "\nProcessing $file"
        python3 ../handle_evaluation_info.py --file "$file"
done

