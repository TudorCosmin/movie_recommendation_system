import pandas as pd
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient

class RecommendationConfig:
    """
    A singleton configuration class for managing resources and settings required for a recommendation system.
    This class ensures that all shared resources (e.g., data, models, clients) are loaded and initialized once.

    Attributes:
        movie_details_df: DataFrame containing details and text descriptions for movies.
        user_details_df: DataFrame containing user details and text descriptions.
        movie_embeddings_df: DataFrame containing precomputed embeddings for movies.
        tokenizer: Tokenizer instance for the pre-trained language model.
        model: Pre-trained language model for generating embeddings or processing text.
        qclient: QdrantClient instance for interacting with the Qdrant database.
        MOVIE_COLLECTION_NAME: Name of the Qdrant collection for storing movie data.
        USER_COLLECTION_NAME: Name of the Qdrant collection for storing user data.
        USER_SEARCH_TOP_K: Number of top results to return for user searches.
        MOVIE_SEARCH_TOP_K: Number of top results to return for movie searches.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RecommendationConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.movie_details_df = pd.read_csv("data/descriptions/movie_text_description.csv")
        self.user_details_df = pd.read_csv("data/descriptions/user_text_description.csv")

        self.movie_embeddings_df = pd.read_csv("data/embeddings/movie_embeddings.csv")

        self.tokenizer = AutoTokenizer.from_pretrained("dunzhang/stella_en_1.5B_v5")
        self.model = AutoModel.from_pretrained("dunzhang/stella_en_1.5B_v5")
        
        self.qclient = QdrantClient(url="http://localhost:6333")
        self.MOVIE_COLLECTION_NAME = "movie_collection"
        self.USER_COLLECTION_NAME = "user_collection"

        self.USER_SEARCH_TOP_K = 25
        self.MOVIE_SEARCH_TOP_K = 10