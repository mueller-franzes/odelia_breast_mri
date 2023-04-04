import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Set the path to your results directory
results_dir = '../results'

# Initialize a dictionary to store AUC values per folder category
auc_dict = defaultdict(lambda: {"values": [], "type": "", "latency": 0})

# Traverse the directory tree
for root, _, files in os.walk(results_dir):
    for file in files:
        if file == 'AUC.txt' or file == 'roc-Malign=1.svg':
            folder_name = os.path.basename(root)
            folder_category = folder_name.split('_')[-1]
            if len(folder_category) == 2:
                folder_category = 'efficientnet_' + folder_category
            if folder_category == 'ResNet34':
                folder_category = 'ResNet50'

            # Read AUC value from the file
            with open(os.path.join(root, file), 'r') as f:
                content = f.read()
                if file == 'AUC.txt':
                    auc_match = re.search(r'ROC \(AUC = (\d+\.\d+)', content)
                else:
                    auc_match = re.search(r'<!-- \(AUC = (\d+\.\d+)\) -->', content)

                if auc_match:
                    auc_value = float(auc_match.group(1))
                    auc_dict[folder_category]["values"].append(auc_value)

            # Add model type and latency information
            if folder_category == 'ResNet50':
                auc_dict[folder_category]["type"] = "3D"
                auc_dict[folder_category]["latency"] = 100  # Add latency value for ResNet50
            elif folder_category == 'att-mil':
                auc_dict[folder_category]["type"] = "2D"
                auc_dict[folder_category]["latency"] = 50  # Add latency value for att-mil

# Sort categories alphabetically and create box chart from AUC values
categories = sorted(auc_dict.keys())
auc_values = [auc_dict[category]["values"] for category in categories]

# Update categories to include model type and latency information
categories = [f"{category} ({auc_dict[category]['type']} model, latency {auc_dict[category]['latency']} ms)" for
              category in categories]

# Choose a colormap for the boxes
cmap = plt.cm.get_cmap("tab10")

fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the size of the figure
bp = ax.boxplot(auc_values, labels=categories)

# Apply colors to the boxes
for i, box in enumerate(bp['boxes']):
    box.set(color=cmap(i % cmap.N), linewidth=2)

# Set plot title and labels
ax.set_title('AUC values per folder category')
ax.set_xlabel('Folder category')
ax.set_ylabel('AUC value')

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Save the plot as an image file
plt.savefig('box_plot.png', dpi=300, bbox_inches='tight')

plt.show()
