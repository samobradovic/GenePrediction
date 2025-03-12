from sklearn.feature_extraction.text import CountVectorizer

k = 3


def get_kmers(seq, k):
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def extract_features(sequences):
    kmers_list = [" ".join(get_kmers(seq, k)) for seq in sequences]

    # Convert k-mers to a numerical feature matrix
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(kmers_list)
    return X
