import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MultiLabelBinarizer


def load_data(filename, names):
    return pd.read_csv(filename, sep='::', encoding='latin-1', header=None, names=names, engine='python')


def main():
    # Setup Movies Dataframe
    movies = load_data('data/movies.dat', ['MovieID', 'Title', 'Genres'])
    movies['Genres'] = movies['Genres'].str.split('|')

    # Setup Users Dataframe
    users = load_data('data/users.dat', ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    users['Zip-code'] = users['Zip-code'].str[0]

    # Setup Ratings Dataframe
    ratings = load_data('data/ratings.dat', ['UserID', 'MovieID', 'Rating', 'Timestamp'])

    # Vectorize genres and use these to cluster the movies
    binarizer = MultiLabelBinarizer()
    feature_vector = binarizer.fit_transform(movies['Genres'])
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(feature_vector)
    movies['Cluster'] = kmeans.predict(feature_vector)

    # Save the movies to a CSV file
    movies.to_csv('movies.csv', index=False)

    # user_gender = input("What is your gender? Type in either \"M\" (male) or \"F\" (female).\n>> ")
    # user_age = input("How old are you? \n>> ")

    user_gender = "M"
    user_age = 18
    zip_code = 8

    # Filter matching users
    users_filtered = users_filtered = users[(users['Gender'] == 'M') & (users['Age'] == 18) & (users['Occupation'] == 16) &
                               (users['Zip-code'] == str(zip_code))]
    while users_filtered.empty:
        if (zip_code + 1) < 10:
            zip_code += 1
        else:
            zip_code = 0

        users_filtered = users[(users['Gender'] == 'M') & (users['Age'] == 18) & (users['Occupation'] == 16) &
                               (users['Zip-code'] == str(zip_code))]

    print(users_filtered)
    ratings_filtered = ratings[ratings['UserID'].isin(users_filtered['UserID']) & (ratings['Rating'] == 5)]
    print(ratings_filtered)
    ratings_filtered = ratings_filtered.drop_duplicates(subset='MovieID', keep="first")

    movies_filtered = movies[movies['MovieID'].isin(ratings_filtered['MovieID'])]
    print(movies_filtered.to_string())


if __name__ == "__main__":
    main()
