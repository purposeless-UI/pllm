#!/bin/bash
# Exit script immediately if any processing block hits an error state
set -e

echo "=============================================================="
# Start your complete processing loop sequentially
echo "🪙 Step 1: Processing text tokens and training BPE codebook..."
python3 src/dataset_pipeline/tokenizer.py

echo "=============================================================="
echo "🚀 Step 2: Launching end-to-end local hardware validation..."
python3 scripts/run_local_test.py

echo "=============================================================="
echo "🎯 System built successfully. Ready for deployment."


##''chmod +x run.sh##
