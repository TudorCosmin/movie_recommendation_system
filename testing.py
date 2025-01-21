from recommendation import recommend_by_movie, recommend_by_user

def main():
    print("\nTesting movie-based recommendation")
    movie_id = 862  # Example: Toy Story
    
    print(f"Finding movies similar to the movie with ID {movie_id}...")
    similar_movies = recommend_by_movie(movie_id=movie_id)
    print("Movies you might like based on your selected movie:")
    print(", ".join(map(str, similar_movies)))

    print("\nTesting user-based recommendation")
    user_id = 1  # Example: User with ID 1
    print(f"Finding recommendations for the user with ID {user_id}...")

    recommended_movies = recommend_by_user(user_id=user_id)
    print("Movies recommended the selected user:")
    print(", ".join(map(str, recommended_movies)))

if __name__ == "__main__":
    main()