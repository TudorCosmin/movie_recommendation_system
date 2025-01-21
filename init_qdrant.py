import time
from qdrant_client import models
import numpy as np
import pandas as pd
from typing import List
from qdrant_client import QdrantClient
from ast import literal_eval

def create_collection(qclient: QdrantClient, collection_name: str, vector_len: int) -> None:
    """
    Create a new collection in Qdrant with the specified name and vector length. 
    If the collection already exists, it is deleted and recreated.

    Args:
        collection_name: The name of the collection to be created.
        vector_len: The length of the vectors that will be stored in this collection.

    Returns:
        None
    """

    qclient.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_len,
            distance=models.Distance.COSINE,
        ),
    )

def prepare_qdrant_points(embedding_df: pd.DataFrame, point_details_df: pd.DataFrame) -> List[models.PointStruct]:
    """
    Prepare points for uploading to a Qdrant collection. Each point consists of an embedding vector 
    and its associated metadata (payload).

    Args:
        embedding_df: DataFrame containing embeddings with an identifier column.
        point_details_df: DataFrame containing detailed metadata for each point.

    Returns:
        A list of points ready to be uploaded to Qdrant.
    """

    points = []
    for idx, row in embedding_df.iterrows():
        details_dict = point_details_df.loc[point_details_df["id"] == row["id"]].iloc[0].to_dict()
        points.append(
            models.PointStruct(
                id=idx,
                vector=np.array(row["embedding"]),
                payload=details_dict
            )
        )
    return points

def upload_points(qclient: QdrantClient, collection_name: str, points: List[models.PointStruct]) -> None:
    """
    Upload a list of points to a specified Qdrant collection.

    Args:
        collection_name: The name of the collection to upload points to.
        points: A list of points to be uploaded.

    Returns:
        None
    """
    
    qclient.upload_points(
        collection_name=collection_name,
        points=points
    )

def initialize_collection(qclient: QdrantClient, collection_name: str, embeddings_filepath: str, details_filepath: str) -> None:
    """
    Initializes a Qdrant collection by creating it (if it does not already exist) and uploading
    data points with their embeddings and metadata.

    Args:
        qclient: An instance of the Qdrant client used to interact with the Qdrant server.
        collection_name: The name of the collection to initialize in Qdrant.
        embeddings_filepath: Path to the CSV file containing embeddings with "id" and "embedding" columns.
        details_filepath: Path to the CSV file containing metadata details for each embedding point.

    Returns:
        None
    """

    if qclient.collection_exists(collection_name):
        print(f"Collection {collection_name} already exists in Qdrant. Skipping...")
        return
    
    init_start_time = time.time()
    print(f"Starting initializing collection {collection_name}...")

    data_df = pd.read_csv(details_filepath)
    embeddings_df = pd.read_csv(embeddings_filepath)
    embeddings_df["embedding"] = embeddings_df["embedding"].apply(literal_eval)
    create_collection(
        qclient=qclient,
        collection_name=collection_name,
        vector_len=len(embeddings_df.iloc[0]["embedding"])
    )
    
    movie_points = prepare_qdrant_points(
        embedding_df=embeddings_df,
        point_details_df=data_df
    )
    
    upload_points(
        qclient=qclient,
        collection_name=collection_name,
        points=movie_points
    )

    init_elapsed_time = time.time() - init_start_time
    print(f"Collection {collection_name} initialization done in {init_elapsed_time:.4f} seconds.\n\n")