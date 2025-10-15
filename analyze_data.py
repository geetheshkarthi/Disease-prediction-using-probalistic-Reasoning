import pandas as pd

# Load the dataset
try:
    df = pd.read_csv("disease_symptoms.csv")
except FileNotFoundError:
    print("Error: disease_symptoms.csv not found in this folder.")
    exit()

# Assume the first column is the disease
disease_col = df.columns[0]
symptom_cols = df.columns[1:]

print(f"Analyzing dataset with disease column: '{disease_col}'\n")

# Melt the dataframe to have one row per disease-symptom pair
df_long = df.melt(id_vars=[disease_col], value_vars=symptom_cols, var_name='symptom_num', value_name='symptom')

# Drop rows where there is no symptom
df_long.dropna(subset=['symptom'], inplace=True)

# Clean up symptom names just like in your training script
df_long['symptom'] = df_long['symptom'].str.strip().str.lower().str.replace('_', ' ')

# Group by symptom and count how many unique diseases it's associated with
symptom_disease_counts = df_long.groupby('symptom')[disease_col].nunique()

# Filter for symptoms that are only associated with ONE disease
perfect_predictors = symptom_disease_counts[symptom_disease_counts == 1]

if not perfect_predictors.empty:
    print(f"🚨 Found {len(perfect_predictors)} symptoms that are 'perfect predictors'.")
    print("These symptoms are only associated with a single disease, making the problem too easy.")
    print("-------------------------------------------------------------------------")
    print(perfect_predictors.to_string())
else:
    print("✅ No obvious 'perfect predictor' symptoms found.")

# Also, check for duplicate rows in the original dataset
num_duplicates = df.duplicated().sum()
if num_duplicates > 0:
    print(f"\n🚨 Warning: Found {num_duplicates} duplicate rows in the dataset.")
else:
    print("\n✅ No duplicate rows found in the dataset.")