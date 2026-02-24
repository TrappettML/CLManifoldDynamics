import os
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm

# --- Configuration ---
DATASET_NAME = "benjamin-paine/imagenet-1k-32x32"
OUTPUT_DIR = "./data/imagenet_28_gray"
IMG_SIZE = 28
NUM_CLASSES = 1000

# Set your target bottlenecks here
TRAIN_SAMPLES_PER_CLASS = 550
TEST_SAMPLES_PER_CLASS = 200
TOTAL_SAMPLES_PER_CLASS = TRAIN_SAMPLES_PER_CLASS + TEST_SAMPLES_PER_CLASS

def process_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define transforms: Grayscale -> Resize (28x28) -> ToTensor
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    print("\nDownloading and loading train split from Hugging Face...")
    # Load ONLY the train split
    dataset = load_dataset(DATASET_NAME, split='train')
    
    total_train_samples = NUM_CLASSES * TRAIN_SAMPLES_PER_CLASS
    total_test_samples = NUM_CLASSES * TEST_SAMPLES_PER_CLASS
    total_target_samples = NUM_CLASSES * TOTAL_SAMPLES_PER_CLASS
    
    print(f"Targeting {TOTAL_SAMPLES_PER_CLASS} samples per class ({total_target_samples} total)...")
    
    # Pre-allocate perfectly sized numpy arrays for both train and test
    X_train = np.zeros((total_train_samples, IMG_SIZE * IMG_SIZE), dtype=np.float32)
    Y_train = np.zeros(total_train_samples, dtype=np.int32)
    
    X_test = np.zeros((total_test_samples, IMG_SIZE * IMG_SIZE), dtype=np.float32)
    Y_test = np.zeros(total_test_samples, dtype=np.int32)
    
    # Track how many samples we've saved for each class
    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    train_saved = 0
    test_saved = 0
    
    print("Processing, grayscaling, and balancing classes...")
    for item in tqdm(dataset, desc="Extracting Train and Test sets"):
        label = item['label']
        
        # Only process if we haven't reached the TOTAL limit for this specific class
        if class_counts[label] < TOTAL_SAMPLES_PER_CLASS:
            img = item['image'].convert("RGB")
            img_tensor = transform(img)  
            img_flat = img_tensor.numpy().flatten()  
            
            # The first 550 go to train, the next 200 go to test
            if class_counts[label] < TRAIN_SAMPLES_PER_CLASS:
                X_train[train_saved] = img_flat
                Y_train[train_saved] = label
                train_saved += 1
            else:
                X_test[test_saved] = img_flat
                Y_test[test_saved] = label
                test_saved += 1
                
            # Increment counters
            class_counts[label] += 1
            
            # Early stop if we have fulfilled the quota for all 1000 classes
            if train_saved + test_saved >= total_target_samples:
                print(f"\nReached target of {total_target_samples} total samples. Stopping early!")
                break
    
    # Safety check
    if train_saved + test_saved < total_target_samples:
        print(f"Warning: Only found {train_saved + test_saved} samples. Some classes lacked {TOTAL_SAMPLES_PER_CLASS} images.")
        X_train, Y_train = X_train[:train_saved], Y_train[:train_saved]
        X_test, Y_test = X_test[:test_saved], Y_test[:test_saved]
        
    print("Saving arrays to disk...")
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "Y_train.npy"), Y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "Y_test.npy"), Y_test)
    print(f"Successfully saved train and test data to {OUTPUT_DIR}/!")

if __name__ == "__main__":
    process_and_save()