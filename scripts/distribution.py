table = '/mnt/sda1/Oliver/segmentation_metadata_unilateral.csv'
import pandas as pd
df = pd.read_csv(table)#, header=[0, 2])
# get the number of ones of third column
print(df[df.columns[2]].value_counts())

table = '/opt/hpe/swarm-learning-hpe/workspace/marugoto_mri/user/data-and-scratch/data/clinical_table.csv'
df = pd.read_csv(table)#, header=[0, 2])
# get the number of ones of third column
print(df[df.columns[1]].value_counts())


# create a mask for patients between 1 and 368
mask_range = df['PATIENT'].str.extract('(\d+)', expand=False).astype(int).between(1, 368)
#get the number of patients between 1 and 368
#print(df.loc[mask_range, 'PATIENT'].nunique())
# count how many are Malign = 1 for the set
count_malign = df.loc[mask_range, 'Malign'].sum()
# count how many are Malign = 0 for the set
count_benign = df.loc[mask_range, 'Malign'].count() - count_malign

print(f"Number of Malign cases from Breast_MRI_001 to Breast_MRI_368: {count_malign}")
print(f"Number of Benign cases from Breast_MRI_001 to Breast_MRI_368: {count_benign}")





# create a mask for patients between 369 and 644
mask_range = df['PATIENT'].str.extract('(\d+)', expand=False).astype(int).between(369, 644)
#get the number of patients between 1 and 368
#print(df.loc[mask_range, 'PATIENT'].nunique())
# count how many are Malign = 1 for the set
count_malign = df.loc[mask_range, 'Malign'].sum()
# count how many are Malign = 0 for the set
count_benign = df.loc[mask_range, 'Malign'].count() - count_malign

print(f"Number of Malign cases from Breast_MRI_369 to Breast_MRI_644: {count_malign}")
print(f"Number of Benign cases from Breast_MRI_369 to Breast_MRI_644: {count_benign}")



# create a mask for patients between 1 and 368
mask_range = df['PATIENT'].str.extract('(\d+)', expand=False).astype(int).between(645, 736)
#get the number of patients between 1 and 368
#print(df.loc[mask_range, 'PATIENT'].nunique())
# count how many are Malign = 1 for the set
count_malign = df.loc[mask_range, 'Malign'].sum()
# count how many are Malign = 0 for the set
count_benign = df.loc[mask_range, 'Malign'].count() - count_malign

print(f"Number of Malign cases from Breast_MRI_645 to Breast_MRI_736: {count_malign}")
print(f"Number of Benign cases from Breast_MRI_645 to Breast_MRI_736: {count_benign}")



# create a mask for patients between 1 and 368
mask_range = df['PATIENT'].str.extract('(\d+)', expand=False).astype(int).between(737, 922)
#get the number of patients between 1 and 368
#print(df.loc[mask_range, 'PATIENT'].nunique())
# count how many are Malign = 1 for the set
count_malign = df.loc[mask_range, 'Malign'].sum()
# count how many are Malign = 0 for the set
count_benign = df.loc[mask_range, 'Malign'].count() - count_malign

print(f"Number of Malign cases from Breast_MRI_737 to Breast_MRI_922: {count_malign}")
print(f"Number of Benign cases from Breast_MRI_737 to Breast_MRI_922: {count_benign}")
