import pandas as pd

df = pd.read_csv("data/processed/processed.csv")

# Calculate the length of each string in the 'Sequence' column
sequence_lengths = df["Sequence"].str.len()

# Print the range of string lengths
print(
    f"Range of string lengths for the 'Sequence' column: {sequence_lengths.min()} - {sequence_lengths.max()}"
)

# Get unique first 3 letters in 'Sequence' column where 'IsGene' column is 1
unique_first_three_letters = df[df["IsGene"] == 1]["Sequence"].str[:3].unique()
print(
    f"Unique first 3 letters in 'Sequence' column where 'IsGene' is 1: {unique_first_three_letters}"
)
