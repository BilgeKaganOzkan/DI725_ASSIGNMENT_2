import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import random
import pandas as pd
import seaborn as sns

# Load annotations
with open('dataset/annotations.json', 'r') as f:
    annotations = json.load(f)

print("Dataset Info:")
print(annotations['info'])

print("\nCategories:")
for i, category in enumerate(annotations['categories']):
    print(f"{i}: {category}")

# Count images and annotations
image_names = set()
for ann in annotations['annotations']:
    image_names.add(ann['image_name'])

print(f"\nTotal unique images: {len(image_names)}")
print(f"Total annotations: {len(annotations['annotations'])}")

# Count objects per category
category_counts = {i: 0 for i in range(len(annotations['categories']))}
for ann in annotations['annotations']:
    for bbox in ann['bbox']:
        category_counts[bbox['class']] += 1

print("\nObjects per category:")
for i, category in enumerate(annotations['categories']):
    print(f"{category}: {category_counts[i]}")

# Save category distribution to CSV
categories = annotations['categories']
counts = [category_counts[i] for i in range(len(categories))]
df = pd.DataFrame({'Category': categories, 'Count': counts})
df.to_csv('dataset_exploration/category_distribution.csv', index=False)

# Create a bar plot of category distribution
plt.figure(figsize=(12, 6))
sns.barplot(x='Category', y='Count', data=df)
plt.title('Number of Objects per Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('dataset_exploration/category_distribution.png')
plt.close()

# Visualize a few examples
def visualize_image(image_name, annotation):
    img_path = os.path.join('dataset/images', image_name)
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]
    
    for bbox in annotation['bbox']:
        left = bbox['left']
        top = bbox['top']
        width = bbox['width']
        height = bbox['height']
        category = bbox['class']
        
        # Draw bounding box
        draw.rectangle([(left, top), (left + width, top + height)], 
                      outline=colors[category], width=3)
        
        # Draw label
        draw.text((left, top - 15), annotations['categories'][category], 
                 fill=colors[category])
    
    return img

# Visualize 5 random images
random.seed(42)
sample_indices = random.sample(range(len(annotations['annotations'])), 5)
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, idx in enumerate(sample_indices):
    ann = annotations['annotations'][idx]
    img = visualize_image(ann['image_name'], ann)
    axes[i].imshow(np.array(img))
    axes[i].set_title(f"Image {i+1}")
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('dataset_exploration/sample_images.png')
plt.close()

# Calculate statistics about bounding boxes
bbox_widths = []
bbox_heights = []
bbox_areas = []
bbox_aspect_ratios = []

for ann in annotations['annotations']:
    for bbox in ann['bbox']:
        width = bbox['width']
        height = bbox['height']
        
        bbox_widths.append(width)
        bbox_heights.append(height)
        bbox_areas.append(width * height)
        bbox_aspect_ratios.append(width / height if height > 0 else 0)

# Create histograms of bounding box properties
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(bbox_widths, bins=50)
axes[0, 0].set_title('Bounding Box Widths')
axes[0, 0].set_xlabel('Width (pixels)')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(bbox_heights, bins=50)
axes[0, 1].set_title('Bounding Box Heights')
axes[0, 1].set_xlabel('Height (pixels)')
axes[0, 1].set_ylabel('Frequency')

axes[1, 0].hist(bbox_areas, bins=50)
axes[1, 0].set_title('Bounding Box Areas')
axes[1, 0].set_xlabel('Area (pixels²)')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(bbox_aspect_ratios, bins=50, range=(0, 5))
axes[1, 1].set_title('Bounding Box Aspect Ratios (width/height)')
axes[1, 1].set_xlabel('Aspect Ratio')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('dataset_exploration/bbox_statistics.png')
plt.close()

# Save summary statistics to a text file
with open('dataset_exploration/dataset_summary.txt', 'w') as f:
    f.write("AU-AIR Dataset Summary\n")
    f.write("=====================\n\n")
    
    f.write("Dataset Info:\n")
    for key, value in annotations['info'].items():
        f.write(f"  {key}: {value}\n")
    
    f.write("\nCategories:\n")
    for i, category in enumerate(annotations['categories']):
        f.write(f"  {i}: {category}\n")
    
    f.write(f"\nTotal unique images: {len(image_names)}\n")
    f.write(f"Total annotations: {len(annotations['annotations'])}\n")
    
    f.write("\nObjects per category:\n")
    for i, category in enumerate(annotations['categories']):
        f.write(f"  {category}: {category_counts[i]}\n")
    
    f.write("\nBounding Box Statistics:\n")
    f.write(f"  Average width: {np.mean(bbox_widths):.2f} pixels\n")
    f.write(f"  Average height: {np.mean(bbox_heights):.2f} pixels\n")
    f.write(f"  Average area: {np.mean(bbox_areas):.2f} pixels²\n")
    f.write(f"  Average aspect ratio: {np.mean(bbox_aspect_ratios):.2f}\n")
    
    f.write(f"  Min width: {np.min(bbox_widths)} pixels\n")
    f.write(f"  Max width: {np.max(bbox_widths)} pixels\n")
    f.write(f"  Min height: {np.min(bbox_heights)} pixels\n")
    f.write(f"  Max height: {np.max(bbox_heights)} pixels\n")

print("Dataset exploration completed. Results saved to dataset_exploration folder.")
