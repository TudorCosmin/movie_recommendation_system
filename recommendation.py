from typing import List
from ast import literal_eval
import numpy as np
from recom_config import RecommendationConfig
from similarity_search import search_similar

config = RecommendationConfig()

def recommend_by_movie(movie_id: int) -> List[int]:
    """
    Recommends similar movies based on the text description of a given movie.

    Args:
        movie_id: The ID of the movie for which to find similar movies.

    Returns:
        A list of IDs of similar movies ranked by relevance.
    """

    movie_text_description = config.movie_details_df.loc[config.movie_details_df["id"] == movie_id].iloc[0]["text"]

    movie_ann = search_similar(
        query=movie_text_description,
        collection_name=config.MOVIE_COLLECTION_NAME,
        top_k=config.MOVIE_SEARCH_TOP_K,
        model=config.model,
        tokenizer=config.tokenizer
    )

    similar_movie_ids = [neighbour.payload["id"] for neighbour in movie_ann.points]

    return similar_movie_ids

def recommend_by_user(user_id: int) -> List[int]:
    """
    Recommends movies to a user based on their preferences and the preferences of similar users.

    Args:
        user_id: The ID of the user for whom to recommend movies.

    Returns:
        A list of IDs of recommended movies, excluding movies the user has already rated.
    """

    row = config.user_details_df.loc[config.user_details_df["id"] == user_id].iloc[0]
    user_text_description = row["text"]
    user_rated_movies = literal_eval(row["favourite_movies"]) + literal_eval(row["mediocre_movies"]) + literal_eval(row["bad_movies"])

    user_ann = search_similar(
        query=user_text_description,
        collection_name=config.USER_COLLECTION_NAME,
        top_k=config.USER_SEARCH_TOP_K,
        model=config.model,
        tokenizer=config.tokenizer
    )

    similar_user_ids = [neighbour.payload["id"] for neighbour in user_ann.points]
    similar_favourite_movies = []
    for id in similar_user_ids:
        fav = literal_eval(config.user_details_df.loc[config.user_details_df["id"] == id].iloc[0]["favourite_movies"])
        similar_favourite_movies.extend(fav)
    
    filtered_df = config.movie_embeddings_df.loc[config.movie_embeddings_df["id"].isin(similar_favourite_movies)]
    filtered_df.loc[:, "embedding"] = filtered_df["embedding"].apply(literal_eval)

    avg_movie_embedding = np.mean(filtered_df["embedding"].tolist(), axis=0).tolist()

    movie_ann = search_similar(
        query=avg_movie_embedding,
        collection_name=config.MOVIE_COLLECTION_NAME,
        top_k=config.MOVIE_SEARCH_TOP_K,
        model=config.model,
        tokenizer=config.tokenizer
    )

    similar_movie_ids = [neighbour.payload["id"] for neighbour in movie_ann.points]
    user_rated_movies
    result = list(filter(lambda x: x not in user_rated_movies, similar_movie_ids))

    return result