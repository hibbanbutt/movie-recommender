
#1.1: Load the MovieLens 1M dataset and extract movie titles
import pandas as pd
#import wikipediaapi
#import wikipedia
import spacy

movies_dat_file = "data\movies.dat"
movies_df = pd.read_csv(movies_dat_file, sep="::", engine="python", header=None, names=["movie_id", "title", "genres"], encoding="ISO-8859-1")

movie_titles = movies_df["title"].tolist()




import requests

#API_KEY = "b3a3c7ea94ff3b67121a83f876500724"  

import tmdbsimple as tmdb
tmdb.API_KEY = "b3a3c7ea94ff3b67121a83f876500724"  

import re

def search_movie(title):
    # Remove the year from the title
    cleaned_title = re.sub(r'\s\(\d{4}\)', '', title)
    
    search = tmdb.Search()
    
    # Search without the year
    response = search.movie(query=cleaned_title)
    if search.results:
        return search.results[0]
    
    # Search with the year included
    response = search.movie(query=title)
    if search.results:
        return search.results[0]
    
    return None


def get_movie_description(movie_id):
    if movie_id is None:
        return None
    movie = tmdb.Movies(movie_id)
    response = movie.info()
    return response["overview"]


movie_ids = [search_movie(title)["id"] if search_movie(title) else None for title in movie_titles]
movie_descriptions = [get_movie_description(movie_id) for movie_id in movie_ids]
movies_df["description"] = movie_descriptions



#Step 2: Text Vectorization using SpaCy
#2.1: Install SpaCy and download the pre-trained model: 
#`pip install spacy
#python -m spacy download en_core_web_lg`
#

#2.2: Load the pre-trained model and process the intro paragraphs

nlp = spacy.load("en_core_web_lg")

def get_document_vector(text):
    if text:
        doc = nlp(text)
        return doc.vector
    else:
        return None

movies_df["intro_vector"] = movies_df["description"].apply(get_document_vector)

# save the DataFrame as a Parquet file:
movies_df.to_parquet("movies_processed.parquet", index=False)



# # Display the first few rows of the DataFrame
# print(movies_df.head())

# # Print the total number of movies
# print(f"Total number of movies: {len(movies_df)}")

# # Check the data types of each column
# print("\nData types of each column:")
# print(movies_df.dtypes)

# # Check for missing values in each column
# print("\nNumber of missing values in each column:")
# print(movies_df.isna().sum())

# # Print titles with missing descriptions
# missing_descriptions = movies_df[movies_df["description"].isna()]
# print("Titles with missing descriptions:")
# print(missing_descriptions["title"])
