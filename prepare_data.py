# prepare_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Specify the path to your CSV or Excel file here
FILE_PATH = "liar_full.csv"  # or "liar_full.xlsx" if it's still in Excel format

# Read the file
if FILE_PATH.endswith('.csv'):
    df = pd.read_csv(FILE_PATH)
elif FILE_PATH.endswith('.xlsx') or FILE_PATH.endswith('.xls'):
    df = pd.read_excel(FILE_PATH)
else:
    raise ValueError("File must be either .csv or .xlsx/.xls")

print(f"Total number of rows: {len(df)}")
print("Columns:", df.columns.tolist())
print("\nSample data:")
print(df.head(2))

# Reorder columns to match the standard LIAR dataset format (important for compatibility with later code)
columns_order = [
    '[ID].json', 'label', 'statement', 'subject(s)', 'speaker', "speaker's job title",
    'state info', 'party affiliation', 'barely true counts', 'false counts',
    'half true counts', 'mostly true counts', 'pants on fire counts', 'venue'
]

# If column names don't exactly match, comment or adjust this line as needed
df = df[columns_order]

# Split data with stratification on 'label' to maintain class balance
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

# Create 'data' folder if it doesn't exist
Path("data").mkdir(exist_ok=True)

# Save as TSV files without header and index (exactly like the original LIAR dataset)
train_df.to_csv("data/train.tsv", sep='\t', index=False, header=False)
valid_df.to_csv("data/valid.tsv", sep='\t', index=False, header=False)
test_df.to_csv("data/test.tsv", sep='\t', index=False, header=False)

print("\nThree files created successfully:")
print(" data/train.tsv")
print(" data/valid.tsv")
print(" data/test.tsv")

print("\nYou can now run main.py!")