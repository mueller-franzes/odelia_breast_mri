import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Set the path to your results directory
results_dir = '/media/swarm/DGX_EXTERNAL/jeff/training_runs_ext_val/'

# Initialize a dictionary to store AUC values per folder category
auc_dict = defaultdict(list)

# Traverse the directory tree
for root, _, files in os.walk(results_dir):
    #get the folder name
    folder = os.path.basename(root)
    #print(folder)
    for file in files:
        if file == 'AUC.txt':
            folder_name = os.path.basename(root)
            folder_category = folder_name.split('_')[-1]
            if len(folder_category) == 2:
                folder_category = 'efficientnet_' + folder_category
            if folder_category == 'ResNet34':
                folder_category = 'ResNet50'
            # Read AUC value from the AUC file
            with open(os.path.join(root, file), 'r') as f:
                content = f.read()
                auc_match = re.search(r'ROC \(AUC = (\d+\.\d+)', content)
                if auc_match:
                    auc_value = float(auc_match.group(1))
                    auc_dict[folder_category].append(auc_value)
print(auc_dict)
# Sort categories alphabetically and create box chart from AUC values
categories = sorted(auc_dict.keys())
auc_values = [auc_dict[category] for category in categories]

# Choose a colormap for the boxes
cmap = plt.cm.get_cmap("tab10")

fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size of the figure
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
