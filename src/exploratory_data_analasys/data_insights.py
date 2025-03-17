import matplotlib.pyplot as plt
import pandas as pd


def get_details_from_csv():
    print("Getting details from the processed CSV file...")
    df = pd.read_csv("data/processed/processed.csv")
    print(df.describe())

    # Calculate the length of each sequence
    df["Sequence_Length"] = df["Sequence"].apply(len)

    # Plot the distribution of sequence lengths
    plt.figure(figsize=(10, 6))
    plt.hist(df["Sequence_Length"], bins=50, edgecolor="black")
    plt.title("Distribution of Sequence Lengths")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.show()

    # Plot the count of gene vs. non-gene sequences
    plt.figure(figsize=(10, 6))
    df["IsGene"].value_counts().plot(kind="bar", color=["blue", "orange"])
    plt.title("Count of Gene vs. Non-Gene Sequences")
    plt.xlabel("IsGene")
    plt.ylabel("Count")
    plt.xticks(ticks=[0, 1], labels=["Non-Gene", "Gene"], rotation=0)
    plt.show()

    # Display basic statistics about the sequences
    print(df.describe())


if __name__ == "__main__":
    get_details_from_csv()
