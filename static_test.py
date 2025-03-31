import os
import numpy as np
import matplotlib.pyplot as plt
from gesture_model import GestureModel

# Set model path
model_path = "/home/ggbo/FYP/Python_code/network_models/finger__TX[0, 1, 2, 3]_RX[0, 1]_posTRAIN[1, 2, 3, 4, 5, 6, 7, 8, 9]_posTEST[1, 2, 3, 4, 5, 6, 7, 8, 9]_bandwidth80convolutional_S1_20epochnetwork.h5"

# Create model instance
gesture_model = GestureModel(model_path)

# Set static and dynamic data directories
static_data_dir = "/home/ggbo/dynamic_dataset/Vmatrices/static/"
dynamic_data_dir = "/home/ggbo/dynamic_dataset/Vmatrices/"

# Get file lists
static_files = [os.path.join(static_data_dir, f) for f in os.listdir(static_data_dir)
if f.endswith('.npy') or f.endswith('.bin')]
dynamic_files = [os.path.join(dynamic_data_dir, f) for f in os.listdir(dynamic_data_dir)
if f.endswith('.npy') or f.endswith('.bin')]

# Collect static indices
static_indices = []
dynamic_indices = []

# Process static files
print("Processing static data...")
for file_path in static_files:
    result = gesture_model.predict_with_static_detection(file_path, verbose=False)
    if result and 'static_index' in result:
        static_indices.append(result['static_index'])
        print(f"{os.path.basename(file_path)}: Static index = {result['static_index']:.6f}")

# Process dynamic files
print("\nProcessing dynamic data...")
for file_path in dynamic_files:
    result = gesture_model.predict_with_static_detection(file_path, verbose=False)
    if result and 'static_index' in result:
        dynamic_indices.append(result['static_index'])
        print(f"{os.path.basename(file_path)}: Static index = {result['static_index']:.6f}")

# Visualize static index distribution
plt.figure(figsize=(10, 6))
plt.hist(static_indices, bins=20, alpha=0.5, label='Static scenes')
plt.hist(dynamic_indices, bins=20, alpha=0.5, label='Dynamic scenes')
plt.axvline(x=0.01, color='r', linestyle='--', label='Current threshold (0.01)')
plt.legend()
plt.xlabel('Static index')
plt.ylabel('Frequency')
plt.title('Static Index Distribution')
plt.savefig('static_index_distribution.png')
plt.show()

# Calculate accuracy at different thresholds
thresholds = np.linspace(0, max(max(static_indices), max(dynamic_indices)), 100)
accuracies = []

for threshold in thresholds:
    # Count of static data correctly classified as static
    correct_static = sum(1 for idx in static_indices if idx < threshold)
    # Count of dynamic data correctly classified as dynamic
    correct_dynamic = sum(1 for idx in dynamic_indices if idx >= threshold)
    
    # Calculate accuracy
    accuracy = (correct_static + correct_dynamic) / (len(static_indices) + len(dynamic_indices))
    accuracies.append(accuracy)

# Find the best threshold
best_threshold = thresholds[np.argmax(accuracies)]
best_accuracy = max(accuracies)

print(f"\nBest threshold: {best_threshold:.6f}")
print(f"Best accuracy: {best_accuracy:.2%}")

# Visualize accuracy vs threshold
plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracies)
plt.axvline(x=best_threshold, color='r', linestyle='--',
            label=f'Best threshold ({best_threshold:.6f})')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Threshold')
plt.legend()
plt.grid(True)
plt.savefig('threshold_accuracy.png')
plt.show()