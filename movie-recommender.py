import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer


def load_data(filename, names):
    return pd.read_csv(filename, sep='::', encoding='latin-1', header=None, names=names, engine='python')


def main():
    # Setup Movies Dataframe
    movies = load_data('data/movies.dat', ['MovieID', 'Title', 'Genres'])
    movies['Genres'] = movies['Genres'].str.split('|')

    # Vectorize genres and use these to cluster the movies
    binarizer = MultiLabelBinarizer()
    feature_vector = binarizer.fit_transform(movies['Genres'])
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(feature_vector)
    movies['Cluster'] = kmeans.predict(feature_vector)

    # Save the movies to a CSV file
    movies.to_csv('movies.csv', index=False)


    print(movies)


if __name__ == "__main__":
    main()
