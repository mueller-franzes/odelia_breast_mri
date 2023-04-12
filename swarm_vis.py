import matplotlib.pyplot as plt
import numpy as np

data = {
    "Swarm": {
        "attmil": [0.71, 0.7, 0.72, 0.72],
        "transformer": [0.74, 0.71, 0.74, 0.76, 0.75],
        "timmViT": [0.76, 0.75, 0.74, 0.75],
        "ResNet": [0.78, 0.72, 0.79],
    },
    "System_1": {
        "attmil": [0.74, 0.71, 0.69],
        "transformer": [0.72, 0.63, 0.68],
        "timmViT": [0.7, 0.69, 0.7],
        "ResNet": [0.69, 0.69],
    },
    "System_2": {
        "attmil": [0.62, 0.62, 0.58],
        "transformer": [0.68, 0.67, 0.71],
        "timmViT": [0.72, 0.66, 0.68],
        "ResNet": [0.69, 0.67],
    },
    "System_3": {
        "attmil": [0.51, 0.46, 0.55],
        "transformer": [0.52, 0.54, 0.46, 0.62, 0.52, 0.62],
        "timmViT": [0.55, 0.52, 0.46],
        "ResNet": [0.63, 0.64],
    },
}

model_types = list(data["Swarm"].keys())
systems = list(data.keys())

grouped_data = []

for model in model_types:
    model_data = []
    for system in systems:
        model_data.append(data[system][model])
    grouped_data.append(model_data)

fig, ax = plt.subplots(figsize=(12, 6))
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

num_models = len(model_types)
num_systems = len(systems)
bar_width = 0.15
spacing = 0.1

for model_idx, model_data in enumerate(grouped_data):
    for system_idx, system_data in enumerate(model_data):
        position = model_idx * (num_systems * bar_width + spacing) + system_idx * bar_width
        ax.boxplot(system_data, positions=[position], widths=bar_width, patch_artist=True, boxprops=dict(facecolor=colors[system_idx]))

ax.set_xticks([(num_systems * bar_width + spacing) / 2 + model_idx * (num_systems * bar_width + spacing) for model_idx in range(num_models)])
ax.set_xticklabels(model_types)
ax.set_xlabel("Model Type")
ax.set_ylabel("Score")

# Create custom legend
custom_lines = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(len(systems))]
ax.legend(custom_lines, systems, loc="upper right")

plt.show()
