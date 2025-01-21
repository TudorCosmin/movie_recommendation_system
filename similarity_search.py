from qdrant_client import models
import torch
from typing import Union, List
from recom_config import RecommendationConfig

config = RecommendationConfig()

def get_embedding(text: str, model, tokenizer) -> List[float]:
    """
    Generate an embedding for a given text using a pre-trained model and tokenizer.

    Args:
        text: The input text to be converted into an embedding.
        model: The pre-trained model used to generate the embedding.
        tokenizer: The tokenizer used to preprocess the text.

    Returns:
        List: The resulting embedding as a list of floats.
    """

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        hard_skill_embedding = torch.mean(embeddings, dim=1).squeeze().numpy()

    return hard_skill_embedding.tolist()

def search_similar(query: Union[str, List[float]], collection_name: str, top_k: int, model, tokenizer) -> List[models.ScoredPoint]:
    """
    Search for similar items in a specified Qdrant collection based on a query, which can be either a text 
    string or an embedding vector.

    Args:
        query: The query input, either a string (to be converted to an embedding) or a list of floats (embedding).
        collection_name: The name of the collection to search in.
        top_k: The number of top similar items to retrieve.
        model: The model used for generating embeddings (if query is a string).
        tokenizer: The tokenizer used for processing the text (if query is a string).

    Returns:
        List: A list of the nearest neighbours based on the query embedding.
    """

    if isinstance(query, str):
        query_emb = get_embedding(text=query, model=model, tokenizer=tokenizer)
    elif isinstance(query, list) and all(isinstance(i, float) for i in query):
        query_emb = query

    nearest_neighbours = config.qclient.query_points(
        collection_name=collection_name,
        query=query_emb,
        limit=top_k
    )

    return nearest_neighbours