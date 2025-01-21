# Recommendation System with Movies Dataset

##  Project Description

This project implements a recommendation system as a robust application designed to provide personalized movie recommendations based on either a specific movie or a user's preferences. It uses the [Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) and Qdrant for vector search and storage.

### Key functionalities

- Movie-Based Recommendations

Given a movie, the system analyzes its attributes and identifies similar movies using a language model and vector-based similarity search. This feature is ideal for users who enjoyed a specific movie and want suggestions for similar content.

- User-Based Recommendations

By analyzing a user's rated movies and preferences (categorized into favorite, mediocre, and bad movies), the system identifies other users with similar tastes. It then uses vector-based similarity search to recommend movies the user is likely to enjoy, excluding those they've already rated.


### Core Components:

- Qdrant Vector Search Database

The system uses Qdrant to manage and search embedding vectors, enabling fast and scalable similarity search across movies and users.

- NLP Transformer Model

[Pre-trained transformer model](https://huggingface.co/dunzhang/stella_en_1.5B_v5) that tokenizes and encodes movie descriptions into high-dimensional embeddings for semantic comparison. It is a very efficient model compared to it's size, check out [this comparison](https://huggingface.co/spaces/mteb/leaderboard).

- Data Integration

It integrates movie metadata, user ratings, and precomputed embeddings, ensuring a seamless flow from raw data to recommendations.


## Running the project

### Prerequisites

1. Python 3.12 installed on your system.
2. Docker installed for running Qdrant.
3. Install Python dependencies wit the following command:
```
pip install -r requirements.txt
```

## Steps to Run the Project

### 1. Download the Dataset

Visit [Movies Dataset on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). Download the dataset and extract all files into the folder `data/initial`.

### 2. Start Qdrant Vector Database

Pull the Qdrant Docker image:

```
docker pull qdrant/qdrant
```

Run the Qdrant Docker container:
```
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```
For more details on setting up Qdrant, check [this page](https://qdrant.tech/documentation/quickstart/).

### 3. Initialize the Data
Run the following command to prepare the data, generate embeddings, and upload points to the Qdrant collection:
```
python init_data.py
```
The script init_data.py performs all necessary initializations, including data preparation, embedding generation, and storing the data in Qdrant.

### 4. Test the Functionalities
Run the following command to test the functionalities of the recommendation system:
```
python testing.py
```
This script will demonstrate the recommendation system's capabilities.

## Notes
- Ensure the Docker container for Qdrant is running while executing the scripts.
- Adjust any file paths in the scripts if your directory structure differs.