import pandas as pd
from pipeline import MovieRecommenderPipeline


def main():
    ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
    movies_df = pd.read_csv('ml-latest-small/movies.csv')
    
    print(f"Ratings: {len(ratings_df)}, Movies: {len(movies_df)}, Users: {ratings_df['userId'].nunique()}")
    
    pipeline = MovieRecommenderPipeline(n_factors=50)
    pipeline.fit(ratings_df, movies_df)
    pipeline.save('recommendation_pipeline.pkl')


if __name__ == "__main__":
    main()