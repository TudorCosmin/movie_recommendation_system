import pandas as pd
import ast
import os
from typing import Any

def safe_eval(value: str, default: str = "[]") -> Any:
    """
    Safely evaluates a JSON-like string into a Python object. If the string is invalid or NaN, 
    returns the provided default value.

    Args:
        value: The string to evaluate.
        default: The default value to return if evaluation fails (default is "[]").

    Returns:
        A Python object parsed from the string or the default value if parsing fails.
    """

    if pd.isna(value):
        return ast.literal_eval(default)
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return ast.literal_eval(default)

def stringify_movie(row: pd.Series) -> str:
    """
    Generates a descriptive text string for a movie based on its metadata.

    Args:
        row: A pandas Series object representing a row from the movie DataFrame.

    Returns:
        A formatted string describing the movie.
    """

    title = row.get("title", "Unknown")

    is_adult = row.get("adult", "Unknown")

    overview = row.get("overview", "No overview available.")

    genres = ", ".join([genre["name"] for genre in safe_eval(row.get("genres", "[]"))])
    genres = genres if genres else "Unknown"

    production_companies = ", ".join([company["name"] for company in safe_eval(row.get("production_companies", "[]"))])
    production_companies = production_companies if production_companies else "Unknown"

    production_countries = ", ".join([country["name"] for country in safe_eval(row.get("production_countries", "[]"))])
    production_countries = production_countries if production_countries else "Unknown"

    spoken_languages = ", ".join([language["name"] for language in safe_eval(row.get("spoken_languages", "[]"))])
    spoken_languages = spoken_languages if spoken_languages else "Unknown"

    movie_string = (
        f"Title: {title}\n"
        f"Genres: {genres}\n"
        f"Overview: {overview}\n"
        f"Adult: {is_adult}\n"
        f"Production Companies: {production_companies}\n"
        f"Production Countries: {production_countries}\n"
        f"Spoken Languages: {spoken_languages}"
    )

    return movie_string

def create_movie_text_description(movies_filepath: str, output_filepath: str) -> None:
    """
    Creates a new CSV file containing movie IDs and their corresponding text descriptions 
    based on metadata. If the output file already exists, the function does nothing.

    Args:
        movies_filepath: Path to the input CSV file containing movie data.
        output_filepath: Path to save the output CSV file with text descriptions.

    Returns:
        None
    """

    if os.path.exists(output_filepath):
        return

    df = pd.read_csv(movies_filepath)

    df["text"] = df.apply(stringify_movie, axis=1)

    output_df = df[["id", "text"]]
    output_df.to_csv(output_filepath, index=False)

    print("Created movie_text_description.csv")

def create_user_text_description(ratings_filepath: str, movies_filepath: str, output_filepath: str) -> None:
    """
    Creates a CSV file containing user-specific text descriptions based on movie ratings.
    Each user is categorized into favorite, mediocre, and bad movies.

    Args:
        ratings_filepath: Path to the CSV file containing user ratings.
        movies_filepath: Path to the CSV file containing movie metadata.
        output_filepath: Path to save the resulting CSV file containing user descriptions.

    Returns:
        None
    """

    if os.path.exists(output_filepath):
        return

    categories_list = [
        ("favourite", 4.0),
        ("mediocre", 2.5),
        ("bad", 0.0)
    ]

    df_meta = pd.read_csv(movies_filepath, usecols=["id", "title"])
    id_to_title = dict(zip(df_meta["id"].astype(str), df_meta["title"]))

    df_ratings = pd.read_csv(ratings_filepath, usecols=["userId", "movieId", "rating"])

    def get_category(r):
        for category, threshold in categories_list:
            if r >= threshold:
                return category
        return "unknown"

    df_ratings["category"] = df_ratings["rating"].apply(get_category)

    grouped_ids = df_ratings.groupby(["userId", "category"])["movieId"].apply(list).unstack(fill_value=[])
    grouped_titles = df_ratings.groupby(["userId", "category"])["movieId"].apply(lambda ids: [id_to_title.get(str(mid)) for mid in ids]).unstack(fill_value=[])

    result = []
    for user_id in grouped_ids.index:
        text_parts = []
        user_data = {"id": user_id}

        for category, _ in categories_list:
            if category in grouped_ids.columns:
                movie_ids = grouped_ids.at[user_id, category]
                movie_titles = grouped_titles.at[user_id, category]

                valid_pairs = [(mid, title) for mid, title in zip(movie_ids, movie_titles) if title]
                first_3_pairs = valid_pairs[:3]

                first_3_ids = [pair[0] for pair in first_3_pairs]
                first_3_titles = ", ".join(pair[1] for pair in first_3_pairs)

                user_data[f"{category}_movies"] = first_3_ids
                text_parts.append(f"{category} movies: {first_3_titles if first_3_titles else "[None]"}")
            else:
                user_data[f"{category}_movies"] = []
                text_parts.append(f"{category} movies: [None]")

        user_data["text"] = "\n".join(text_parts)
        result.append(user_data)

    result_df = pd.DataFrame(result)
    result_df.to_csv(output_filepath, index=False)