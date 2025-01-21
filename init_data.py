import os
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient

from init_cleaning import clean_movie_data
from init_descriptions import create_movie_text_description, create_user_text_description
from init_embeddings import generate_embeddings
from init_qdrant import initialize_collection

def ensure_folder_structure(base_path="data"):
    subfolders = ["cleaned", "descriptions", "embeddings", "initial"]

    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    for subfolder in subfolders:
        folder_path = os.path.join(base_path, subfolder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

def main():
    ensure_folder_structure()

    clean_movie_data(
        movies_filepath="data/initial/movies_metadata.csv",
        output_filepath="data/cleaned/movies_metadata.csv"
    )

    create_movie_text_description(
        movies_filepath="data/cleaned/movies_metadata.csv",
        output_filepath="data/descriptions/movie_text_description.csv"
    )
    create_user_text_description(
        ratings_filepath="data/initial/ratings.csv",
        movies_filepath="data/cleaned/movies_metadata.csv",
        output_filepath="data/descriptions/user_text_description.csv"
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("dunzhang/stella_en_1.5B_v5")
    model = AutoModel.from_pretrained("dunzhang/stella_en_1.5B_v5")
    print(f"Using device: {device}")

    generate_embeddings(
        input_filepath="data/descriptions/movie_text_description.csv",
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_filepath="data/embeddings/movie_embeddings.csv",
        max_limit=50000
    )
    generate_embeddings(
        input_filepath="data/descriptions/user_text_description.csv",
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_filepath="data/embeddings/user_embeddings.csv",
        max_limit=50000
    )

    # https://qdrant.tech/documentation/quickstart/
    qclient = QdrantClient(url="http://localhost:6333")

    initialize_collection(
        qclient=qclient,
        collection_name="movie_collection",
        embeddings_filepath="data/embeddings/movie_embeddings.csv",
        details_filepath="data/descriptions/movie_text_description.csv"
    )
    initialize_collection(
        qclient=qclient,
        collection_name="user_collection",
        embeddings_filepath="data/embeddings/user_embeddings.csv",
        details_filepath="data/descriptions/user_text_description.csv"
    )

if __name__ == "__main__":
    main()