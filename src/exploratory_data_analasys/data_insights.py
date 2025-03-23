import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def get_all_start_codons():
    """Finds all unique 3-letter start codons where IsGene == 1 and prints their
    occurrences and frequencies."""
    df = pd.read_csv("data/processed/processed.csv")

    # Filter rows where IsGene == 1 and extract the first 3 letters of the Sequence
    # column
    start_codons = df[df["IsGene"] == 1]["Sequence"].str[:3]

    # Get the frequency count of each unique 3-letter sequence
    codon_counts = start_codons.value_counts()

    # Set pandas options to display all rows
    pd.set_option("display.max_rows", None)  # Show all rows
    pd.set_option("display.max_columns", None)  # Show all columns (if needed)
    pd.set_option(
        "display.expand_frame_repr", False
    )  # Prevent wrapping to multiple lines

    print(
        "Unique first 3 letters in 'Sequence' column where 'IsGene' is 1 with their "
        "frequencies:"
    )
    print(codon_counts)

    # Reset pandas options to default after printing
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")
    pd.reset_option("display.expand_frame_repr")


def get_all_stop_codons():
    """Finds all unique 3-letter start codons where IsGene == 1 and prints their
    occurrences and frequencies."""
    df = pd.read_csv("data/processed/processed.csv")

    # Filter rows where IsGene == 1 and extract the last 3 letters
    # of the Sequence column
    stop_codons = df[df["IsGene"] == 1]["Sequence"].str[-3:]

    # Get the frequency count of each unique 3-letter sequence
    codon_counts = stop_codons.value_counts()

    # Set pandas options to display all rows
    pd.set_option("display.max_rows", None)  # Show all rows
    pd.set_option("display.max_columns", None)  # Show all columns (if needed)
    pd.set_option(
        "display.expand_frame_repr", False
    )  # Prevent wrapping to multiple lines

    print(
        "Unique last 3 letters in 'Sequence' column where 'IsGene' is 1 with their "
        "frequencies:"
    )
    print(codon_counts)

    # Reset pandas options to default after printing
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")
    pd.reset_option("display.expand_frame_repr")


def correlation_heatmap():
    """Generates a heatmap of the correlation matrix of the processed CSV file."""
    df = pd.read_csv("data/processed/processed.csv")

    # Calculate the correlation matrix
    corr = df.corr()

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()


if __name__ == "__main__":
    # get_details_from_csv()
    # get_all_start_codons()
    # get_all_stop_codons()
    correlation_heatmap()
