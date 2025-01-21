import torch
import os
import csv
import json
import pandas as pd
from tqdm import tqdm
from typing import List

def get_embedding(text: str, model, tokenizer, device: str) -> List[float]:
    """
    Generates a numerical embedding for a given text using a specified language model 
    and tokenizer. The embedding is computed as the mean of the token embeddings.

    Args:
        text: The input text to embed.
        model: The model used for generating embeddings.
        tokenizer: The tokenizer corresponding to the model.
        device: The device ("cpu" or "mps") to run the computation on.

    Returns:
        A list of floats representing the embedding of the input text.
    """

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        output_embedding = torch.mean(embeddings, dim=1).squeeze().cpu().numpy()

    return output_embedding.tolist()

def generate_embeddings(input_filepath: str, model, tokenizer, device: str, output_filepath: str, max_limit: int = 50000) -> None:
    """
    Generates text embeddings for input data using a specified model and tokenizer, 
    and appends the embeddings to an output CSV file. The function avoids duplicating 
    already processed data and ensures the total number of records does not exceed the 
    specified limit.

    Args:
        input_filepath: Path to the input CSV file containing data with "id" and "text" columns.
        model: The trained model used for generating embeddings.
        tokenizer: The tokenizer corresponding to the trained model.
        device: The device to run the computation on.
        output_filepath: Path to the CSV file where embeddings will be stored. 
                         The file is created if it does not exist.
        max_limit: The maximum number of records allowed in the output file. Defaults to 50,000.

    Returns:
        None
    """

    model = model.to(device)

    input_df = pd.read_csv(input_filepath)
    input_ls = input_df.to_dict("records")

    try:
        existing_df = pd.read_csv(output_filepath)
        existing_elements = set(existing_df["id"])

        if len(existing_elements) >= max_limit:
            print(f"Output file already contains {max_limit} or more records. No further processing required.")
            return
    except FileNotFoundError:
        existing_elements = set()

    new_elements_ls = [elem for elem in input_ls if elem["id"] not in existing_elements]

    available_slots = max_limit - len(existing_elements)
    if available_slots <= 0:
        print(f"No available slots left to process. {output_filepath} has reached the limit of {max_limit} records.")
        return

    new_elements_ls = new_elements_ls[:available_slots]
    total_new_elements = len(new_elements_ls)

    if total_new_elements == 0:
        print(f"No new elements to process. {output_filepath} is already up-to-date.")
        return

    with open(output_filepath, "a", newline="", encoding="utf-8") as f_output:
        writer = csv.writer(f_output, quoting=csv.QUOTE_MINIMAL)

        if f_output.tell() == 0:
            writer.writerow(["id", "embedding"])

        with tqdm(total=total_new_elements, desc=f"Processing {os.path.basename(output_filepath)}") as pbar:
            for elem in new_elements_ls:
                try:
                    text = elem["text"]
                    embedding_list = get_embedding(text, model, tokenizer, device)
                    embedding_str = json.dumps(embedding_list)

                    writer.writerow([elem["id"], embedding_str])
                except Exception as e:
                    print(f"Error processing element ID {elem["id"]}: {e}")
                finally:
                    pbar.update(1)

    print(f"Embeddings generation completed for {total_new_elements} new elements.")
