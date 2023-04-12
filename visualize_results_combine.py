import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Set the path to your results directory
results_dir_mil = '/home/jeff/PycharmProjects/marugoto/results'
results_dir = './results'
results_run = "/home/jeff/PycharmProjects/odelia_breast_mri/prediction_runs"
# Initialize a dictionary to store AUC values per folder category
auc_dict = defaultdict(list)

# Traverse the directory tree for MIL 2D methods
for root, _, files in os.walk(results_dir_mil):
    for file in files:
        if file == 'roc-Malign=1.svg':
            folder_name = os.path.basename(root)
            folder_category = '_'.join(folder_name.split('_')[:-1])
            if folder_category == "vit_sophia":
                folder_category = "vit_mil"
            if folder_category == "mil_2a":
                folder_category = "att_2layer_mil"
            if folder_category == "timmViT":
                folder_category = "vit_lstm_mil"
            if folder_category == "timmVit_imrove":
                folder_category = "vit_lstm_deeper_mil"
            if folder_category == "vit_spatial":
                folder_category = "vit_spatial_mil"
            # Read AUC value from the SVG file
            with open(os.path.join(root, file), 'r') as f:
                content = f.read()
                auc_match = re.search(r'<!-- \(AUC = (\d+\.\d+)\) -->', content)
                if auc_match:
                    auc_value = float(auc_match.group(1))
                    auc_dict[folder_category].append(auc_value)

sorted_auc_dict1 = {k: auc_dict[k] for k in sorted(auc_dict)}
auc_dict = defaultdict(list)
# Traverse the directory tree for 3D methods
for root, _, files in os.walk(results_dir):
    for file in files:
        if file == 'AUC.txt':
            folder_name = os.path.basename(root)
            folder_category = folder_name.split('_')[-1]
            if len(folder_category) == 2:
                folder_category = 'efficientnet_' + folder_category
            if folder_category == "ResNet34":
                folder_category = "ResNet50"

            # Read AUC value from the AUC file
            with open(os.path.join(root, file), 'r') as f:
                content = f.read()
                auc_match = re.search(r'ROC \(AUC = (\d+\.\d+)', content)
                if auc_match:
                    auc_value = float(auc_match.group(1))
                    auc_dict[folder_category].append(auc_value)
auc_dict['UNet3D'].append(0.66)
auc_dict['UNet3D'].append(0.69)
auc_dict['UNet3D'].append(0.63)

sorted_auc_dict2 = {k: auc_dict[k] for k in sorted(auc_dict)}
sorted_auc_dict1.update(sorted_auc_dict2)
# Sort categories alphabetically and create box chart from AUC values
#categories = sorted(auc_dict.keys())
categories=sorted_auc_dict1.keys()
auc_values = [sorted_auc_dict1[category] for category in categories]
for key,value in sorted_auc_dict1.items():
    print (key, value)
# Color mapping for similar folder categories
color_mapping = {
    "ResNet": "tab:blue",
    "efficientnet": "tab:orange",
    "EfficientNet3D": "tab:red",
    "vit": "tab:purple",
    "DenseNet": "tab:brown",
    "other": "tab:green",
    "attention": "tab:cyan",
    "3d_cnn": "tab:blue",
    "2d_cnn": "tab:orange"
}

# Choose a colormap for the boxes
cmap = plt.cm.get_cmap("tab10")

fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the size of the figure
bp = ax.boxplot(auc_values, labels=categories)
attention_based_models = ["att_2layer_mil", "att_mil", "vit_lstm_deeper_mil", "vit_lstm_mil", "vit_lstm_wo_1d", "vit_mil", "vit_spatial_mil"]
attention_indices = [i for i, category in enumerate(categories) if category in attention_based_models]

# 2. Add vertical lines to the plot
plt.axvline(x=attention_indices[-1] + 1.5, linestyle='--', color='r', alpha=0.5)
# Apply colors to the boxes
for i, box in enumerate(bp['boxes']):
    category = list(categories)[i]
    color_key = None

    if category in attention_based_models:
        color_key = "attention"
    elif category in ["DenseNet121", "EfficientNet3Db7", "ResNet101", "ResNet152", "ResNet18", "ResNet50", "UNet3D"]:
        color_key = "3d_cnn"
    elif category in ["efficientnet_b4", "efficientnet_b7"]:
        color_key = "2d_cnn"
    else:
        color_key = "other"

    box.set(color=color_mapping[color_key], linewidth=2)

# Set plot title and labels
ax.set_title('Benchmark on DUKE radiology MRI dataset')
ax.set_xlabel('Model Name')
ax.set_ylabel('AUC value')

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')


# 1. Determine the index of the models that are based on attention mechanisms
attention_based_models = ["vit_spatial_mil"]
attention_indices = [i for i, category in enumerate(categories) if category in attention_based_models]

# 2. Add vertical lines to the plot
for index in attention_indices:
    plt.axvline(x=index + 1.5, linestyle='--', color='r', alpha=0.5)

# 3. Update the legend to include information about the attention-based models
legend_elements = [
    plt.Line2D([0], [0], linestyle='--', color='r', label='Division between models') , # New legend entry

    plt.Line2D([0], [0], color=color_mapping["attention"], label='Attention-based models', linewidth=2),
    plt.Line2D([0], [0], color=color_mapping["3d_cnn"], label='3D CNN models', linewidth=2),
    plt.Line2D([0], [0], color=color_mapping["2d_cnn"], label='2D CNN models', linewidth=2)
]

ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), frameon=True)

# Save the plot as an image file
plt.savefig('box_plot.png', dpi=300, bbox_inches='tight')

plt.show()




