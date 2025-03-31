import logging

import torch
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True
)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

# Example DNA sequence
sequence = "ATGCGTACGTAGCTAGCTAGCTAGC"

# Tokenize the sequence
tokens = tokenizer(sequence, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**tokens)

# Print the outputs to understand its structure
print(outputs)

# Assuming the first element of the tuple is the logits
logits = outputs[0]

# Get the predicted class for each token
predictions = torch.argmax(logits, dim=-1)

# Log the predictions
# logging.info(predictions)

# # Map predictions back to the input sequence
# token_ids = tokens["input_ids"][0].tolist()
# predicted_labels = predictions[0].tolist()

# # Decode the tokens and print the sequence with predicted labels
# decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids)
# for token, label in zip(decoded_tokens, predicted_labels):
#     print(f"Token: {token}, Label: {label}")
