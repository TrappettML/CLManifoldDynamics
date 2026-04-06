#!/bin/bash

# Ensure an argument is passed
if [ -z "$1" ]; then
    echo "Error: No array index provided."
    exit 1
fi

ARRAY_INDEX=$1

# 1. Define your parameter search space
# Add multiple values to search over them. Leave a single value to keep it constant.
ALGORITHMS=("SL")
LR1_VALS=("1e-2" "1e-3" "1e-4")
LR2_VALS=("1e-4" "1e-5")
NUM_TASKS=("5") # Constant for this particular sweep

# 2. Initialize empty arrays to hold the exact combinations
COMBINED_ALG=()
COMBINED_LR1=()
COMBINED_LR2=()
COMBINED_TASKS=()

# 3. Generate all combinations (Cartesian Product)
for alg in "${ALGORITHMS[@]}"; do
    for lr1 in "${LR1_VALS[@]}"; do
        for lr2 in "${LR2_VALS[@]}"; do
            for task in "${NUM_TASKS[@]}"; do
                COMBINED_ALG+=("$alg")
                COMBINED_LR1+=("$lr1")
                COMBINED_LR2+=("$lr2")
                COMBINED_TASKS+=("$task")
            done
        done
    done
done

# Calculate total generated combinations
TOTAL_COMBOS=${#COMBINED_ALG[@]}

# 4. Out-of-bounds safety check
if [ "$ARRAY_INDEX" -ge "$TOTAL_COMBOS" ] || [ "$ARRAY_INDEX" -lt 0 ]; then
    echo "Error: Slurm Array index $ARRAY_INDEX is out of bounds."
    echo "Only $TOTAL_COMBOS combinations were generated (valid indices: 0 to $((TOTAL_COMBOS-1)))."
    exit 1
fi

# 5. Extract the specific parameters for this job index
ALG=${COMBINED_ALG[$ARRAY_INDEX]}
LR1=${COMBINED_LR1[$ARRAY_INDEX]}
LR2=${COMBINED_LR2[$ARRAY_INDEX]}
TASKS=${COMBINED_TASKS[$ARRAY_INDEX]}

echo "=========================================================="
echo "Grid Search Configuration"
echo "Array Index: $ARRAY_INDEX / $((TOTAL_COMBOS-1))"
echo "Algorithm:   $ALG"
echo "LR1:         $LR1"
echo "LR2:         $LR2"
echo "Num Tasks:   $TASKS"
echo "=========================================================="

# 6. Execute the Python script
python single_run.py \
    --algorithm "$ALG" \
    --lr1 "$LR1" \
    --lr2 "$LR2" \
    --num_tasks "$TASKS" \
    --num_epochs 100