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
TRAIN_SAMPLES_PER_CLASS = 600
TEST_SAMPLES_PER_CLASS = 50

def process_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define transforms: Grayscale -> Resize (28x28) -> ToTensor
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    # Structure: (HuggingFace Split, Output Name, Target Samples Per Class)
    splits = [
        ('train', 'train', TRAIN_SAMPLES_PER_CLASS), 
        ('validation', 'test', TEST_SAMPLES_PER_CLASS)
    ]
    
    for hf_split, out_name, samples_per_class in splits:
        print(f"\nDownloading and loading {hf_split} split from Hugging Face...")
        dataset = load_dataset(DATASET_NAME, split=hf_split)
        
        total_target_samples = NUM_CLASSES * samples_per_class
        print(f"Targeting {samples_per_class} samples per class ({total_target_samples} total)...")
        
        # Pre-allocate perfectly sized numpy arrays
        X_all = np.zeros((total_target_samples, IMG_SIZE * IMG_SIZE), dtype=np.float32)
        Y_all = np.zeros(total_target_samples, dtype=np.int32)
        
        # Track how many samples we've saved for each class
        class_counts = {i: 0 for i in range(NUM_CLASSES)}
        saved_count = 0
        
        print(f"Processing, grayscaling, and balancing classes...")
        for item in tqdm(dataset, desc=f"Processing {out_name}"):
            label = item['label']
            
            # Only process if we haven't reached the limit for this specific class
            if class_counts[label] < samples_per_class:
                img = item['image'].convert("RGB")
                img_tensor = transform(img)  
                img_flat = img_tensor.numpy().flatten()  
                
                # Save to array
                X_all[saved_count] = img_flat
                Y_all[saved_count] = label
                
                # Increment counters
                class_counts[label] += 1
                saved_count += 1
                
                # Early stop if we have fulfilled the quota for all 1000 classes
                if saved_count >= total_target_samples:
                    print(f"\nReached target of {total_target_samples} total samples. Stopping early!")
                    break
        
        # Safety check in case the dataset didn't have enough samples for some classes
        if saved_count < total_target_samples:
            print(f"Warning: Only found {saved_count} samples. Some classes have fewer than {samples_per_class}.")
            X_all = X_all[:saved_count]
            Y_all = Y_all[:saved_count]
            
        # Save directly to highly-optimized .npy format
        x_path = os.path.join(OUTPUT_DIR, f"X_{out_name}.npy")
        y_path = os.path.join(OUTPUT_DIR, f"Y_{out_name}.npy")
        
        print(f"Saving arrays to disk...")
        np.save(x_path, X_all)
        np.save(y_path, Y_all)
        print(f"Successfully saved {out_name} data to {OUTPUT_DIR}/!")

if __name__ == "__main__":
    process_and_save()