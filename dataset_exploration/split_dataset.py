import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Create metadata directory if it doesn't exist
os.makedirs('metadata', exist_ok=True)

# Load annotations
print("Loading annotations...")
with open('dataset/annotations.json', 'r') as f:
    annotations = json.load(f)

# Get all unique image names
image_names = set()
for ann in annotations['annotations']:
    image_names.add(ann['image_name'])

image_list = list(image_names)
print(f"Total unique images: {len(image_list)}")

# Count objects per category in each image
image_category_counts = {}
for ann in annotations['annotations']:
    image_name = ann['image_name']
    category_counts = {i: 0 for i in range(len(annotations['categories']))}

    for bbox in ann['bbox']:
        category_counts[bbox['class']] += 1

    image_category_counts[image_name] = category_counts

# Split dataset into train, validation, and test sets (70%, 15%, 15%)
# First split into train and temp
train_images, temp_images = train_test_split(image_list, test_size=0.3, random_state=42)

# Then split temp into validation and test
val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

print(f"Split dataset: {len(train_images)} training images, {len(val_images)} validation images, {len(test_images)} test images")

# Verify that the splits maintain similar class distributions
def calculate_distribution(image_subset):
    distribution = {i: 0 for i in range(len(annotations['categories']))}
    for img in image_subset:
        for category, count in image_category_counts[img].items():
            distribution[category] += count
    return distribution

train_dist = calculate_distribution(train_images)
val_dist = calculate_distribution(val_images)
test_dist = calculate_distribution(test_images)

# Normalize distributions for comparison
total_train = sum(train_dist.values())
total_val = sum(val_dist.values())
total_test = sum(test_dist.values())

train_dist_norm = {k: v/total_train for k, v in train_dist.items()}
val_dist_norm = {k: v/total_val for k, v in val_dist.items()}
test_dist_norm = {k: v/total_test for k, v in test_dist.items()}

# Create a dataframe for visualization
categories = annotations['categories']
dist_df = pd.DataFrame({
    'Category': categories,
    'Train': [train_dist[i] for i in range(len(categories))],
    'Validation': [val_dist[i] for i in range(len(categories))],
    'Test': [test_dist[i] for i in range(len(categories))]
})

# Save distribution to CSV
dist_df.to_csv('dataset_exploration/split_distribution.csv', index=False)

# Plot the distribution
plt.figure(figsize=(12, 6))
x = np.arange(len(categories))
width = 0.25

plt.bar(x - width, [train_dist_norm[i] for i in range(len(categories))], width, label='Train')
plt.bar(x, [val_dist_norm[i] for i in range(len(categories))], width, label='Validation')
plt.bar(x + width, [test_dist_norm[i] for i in range(len(categories))], width, label='Test')

plt.xlabel('Category')
plt.ylabel('Normalized Count')
plt.title('Category Distribution Across Splits')
plt.xticks(x, categories, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('dataset_exploration/split_distribution.png')
plt.close()

# Create train, validation, and test annotation files
train_annotations = {
    'info': annotations['info'],
    'licenses': annotations['licenses'],
    'categories': annotations['categories'],
    'annotations': []
}

val_annotations = {
    'info': annotations['info'],
    'licenses': annotations['licenses'],
    'categories': annotations['categories'],
    'annotations': []
}

test_annotations = {
    'info': annotations['info'],
    'licenses': annotations['licenses'],
    'categories': annotations['categories'],
    'annotations': []
}

for ann in annotations['annotations']:
    if ann['image_name'] in train_images:
        train_annotations['annotations'].append(ann)
    elif ann['image_name'] in val_images:
        val_annotations['annotations'].append(ann)
    elif ann['image_name'] in test_images:
        test_annotations['annotations'].append(ann)

# Save the split annotations to metadata folder
print("Saving annotation files...")
with open('metadata/train_annotations.json', 'w') as f:
    json.dump(train_annotations, f)

with open('metadata/val_annotations.json', 'w') as f:
    json.dump(val_annotations, f)

with open('metadata/test_annotations.json', 'w') as f:
    json.dump(test_annotations, f)

# Save the image lists for reference
with open('metadata/train_images.txt', 'w') as f:
    for img in train_images:
        f.write(f"{img}\n")

with open('metadata/val_images.txt', 'w') as f:
    for img in val_images:
        f.write(f"{img}\n")

with open('metadata/test_images.txt', 'w') as f:
    for img in test_images:
        f.write(f"{img}\n")

print("Dataset splitting completed. Annotation files saved to metadata folder.")
print("Distribution visualization saved to dataset_exploration folder.")
