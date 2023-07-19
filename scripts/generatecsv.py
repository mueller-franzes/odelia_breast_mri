import glob

import pandas as pd
import numpy as np

# Define the mapping function
def map_file_name_to_model_name(file_name):
    file_name = file_name.lower()
    if "resnet50-2d" in file_name:
        return "ResNet50-2D"
    # elif "resnet50" in file_name but it is not resnet50-2d then return "ResNet50-3D"
    if "resnet50" in file_name and "resnet50-2d" not in file_name:
        return "ResNet50-3D"
    elif "resnet101" in file_name:
        return "ResNet101-3D"
    elif "resnet18" in file_name:
        return "ResNet18-3D"
    elif "densnet121" in file_name:
        return "DenseNet121-3D"
    elif "densenet121" in file_name:
        return "DenseNet121-3D"
    elif "timmvit" in file_name:
        return "ViT-LSTM-MIL"
    elif "transformer" in file_name:
        return "ViT-MIL"
    elif "attmil" in file_name:
        return "Att-MIL"
    else:
        return None  # Return None for unknown models

pathname = '/mnt/sda1/Duke Compare/delongtest/ext_merge/'
# get the file under the folder with delong_test in it
file = glob.glob(pathname+'*delong_test*')
print(file)
# Load the data
if len(file) == 1:
    data = pd.read_csv(file[0], delimiter='\t')

# Process the data to extract the model pairs and p-values
model_pairs = []
p_values = []
model1 = model2 = None
for i, row in data.iterrows():
    if "p-value" in row.values[0]:
        p_value = float(row.values[0].split()[2])
        if model1 is not None and model2 is not None:
            model_pairs.append((model1, model2))
            p_values.append(p_value)
    else:
        model1 = map_file_name_to_model_name(row.values[0].split(" vs ")[0])
        model2 = map_file_name_to_model_name(row.values[0].split(" vs ")[1])

# Define the model names
model_names = ['ResNet18-3D', 'ResNet50-3D', 'ResNet101-3D', 'DenseNet121-3D', 'ViT-MIL', 'ViT-LSTM-MIL', 'Att-MIL', 'ResNet50-2D']

# Create a DataFrame for the comparison matrix
df = pd.DataFrame(np.zeros((len(model_names), len(model_names))), columns=model_names, index=model_names)

# Fill the diagonal with 1s
np.fill_diagonal(df.values, 1)

# Fill in the comparison results
for i in range(len(model_pairs)):
    model1, model2 = model_pairs[i]
    p_value = p_values[i]
    if model1 in model_names and model2 in model_names:
        df.loc[model1, model2] = p_value
        df.loc[model2, model1] = p_value

# Save the DataFrame to a CSV file, rounding to 3 decimal places
df.round(3).to_csv(pathname+'delong_test.csv')
