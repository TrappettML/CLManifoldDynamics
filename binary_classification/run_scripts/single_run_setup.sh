#!/bin/bash

# ============================================
# CONFIGURATION SWITCH
# ============================================
MODE="list"   # grid or list
# ============================================

if [ -z "$1" ]; then
    echo "Error: No array index provided."
    exit 1
fi
ARRAY_INDEX=$1

# Parameter values (used only in "grid" mode)
ALGORITHMS=("SL")
LR1_VALS=("1e-1" "1e-2" "1e-3" "1e-4")
LR2_VALS=("1e-1" "1e-2" "1e-3" "1e-4")
NUM_TASKS=("20")
NUM_EPOCHS=("1000")
NUM_DIM_OUT=("10")

# Initialize arrays
COMBINED_ALG=()
COMBINED_LR1=()
COMBINED_LR2=()
COMBINED_TASKS=()
COMBINED_EPOCHS=()
COMBINED_D_OUT=()

if [ "$MODE" = "grid" ]; then
    # --- Cartesian product generation ---
    for alg in "${ALGORITHMS[@]}"; do
        for lr1 in "${LR1_VALS[@]}"; do
            for lr2 in "${LR2_VALS[@]}"; do
                for task in "${NUM_TASKS[@]}"; do
                    for epoch in "${NUM_EPOCHS[@]}"; do
                        for d_out in "${NUM_DIM_OUT[@]}"; do
                            COMBINED_ALG+=("$alg")
                            COMBINED_LR1+=("$lr1")
                            COMBINED_LR2+=("$lr2")
                            COMBINED_TASKS+=("$task")
                            COMBINED_EPOCHS+=("$epoch")
                            COMBINED_D_OUT+=("$d_out")
                        done
                    done
                done
            done
        done
    done

elif [ "$MODE" = "list" ]; then
    # --- Explicit list of combinations ---
    # Format: "alg lr1 lr2 tasks epochs d_out"
    COMBOS=(
        "SL 1e-4 1e-2 20 1000 1"
        "SL 1e-4 1e-1 20 1000 1"
        "SL 1e-1 1e-1 20 1000 1"
        "SL 1e-1 1e-4 20 1000 1"
        "SL 1e-1 1e-1 20 1000 2"
        "SL 1e-1 1e-2 20 1000 2"
        "SL 1e-1 1e-3 20 1000 2"
        "SL 1e-1 1e-4 20 1000 2"
        "SL 1e-3 1e-4 20 1000 3"
        "SL 1e-3 1e-1 20 1000 3"
        "SL 1e-2 1e-1 20 1000 3"
    )

    for combo in "${COMBOS[@]}"; do
        # Split the string into individual variables
        read -r alg lr1 lr2 tasks epochs d_out <<< "$combo"
        COMBINED_ALG+=("$alg")
        COMBINED_LR1+=("$lr1")
        COMBINED_LR2+=("$lr2")
        COMBINED_TASKS+=("$tasks")
        COMBINED_EPOCHS+=("$epochs")
        COMBINED_D_OUT+=("$d_out")
    done

else
    echo "Error: Unknown MODE '$MODE'. Use 'grid' or 'list'."
    exit 1
fi

# Rest of the script remains identical
TOTAL_COMBOS=${#COMBINED_ALG[@]}

if [ "$ARRAY_INDEX" -ge "$TOTAL_COMBOS" ] || [ "$ARRAY_INDEX" -lt 0 ]; then
    echo "Error: SLURM array index $ARRAY_INDEX out of bounds (0..$((TOTAL_COMBOS-1)))."
    exit 1
fi

ALG=${COMBINED_ALG[$ARRAY_INDEX]}
LR1=${COMBINED_LR1[$ARRAY_INDEX]}
LR2=${COMBINED_LR2[$ARRAY_INDEX]}
TASKS=${COMBINED_TASKS[$ARRAY_INDEX]}
EPOCHS=${COMBINED_EPOCHS[$ARRAY_INDEX]}
D_OUT=${COMBINED_D_OUT[$ARRAY_INDEX]}

echo "=========================================================="
echo "Configuration (Mode: $MODE)"
echo "Array Index: $ARRAY_INDEX / $((TOTAL_COMBOS-1))"
echo "Algorithm:   $ALG"
echo "LR1:         $LR1"
echo "LR2:         $LR2"
echo "Num Tasks:   $TASKS"
echo "Num Epochs:  $EPOCHS"
echo "Dim Out:     $D_OUT"
echo "=========================================================="

python single_run.py \
    --algorithm "$ALG" \
    --lr1 "$LR1" \
    --lr2 "$LR2" \
    --num_tasks "$TASKS" \
    --num_epochs "$EPOCHS" \
    --out_dim "$D_OUT"