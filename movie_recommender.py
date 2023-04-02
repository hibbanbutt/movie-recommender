import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import MultiLabelBinarizer


def load_data(filename, names):
    return pd.read_csv(filename, sep='::', encoding='latin-1', header=None, names=names, engine='python')


def main():
    # Setup Ratings Dataframe
    ratings = load_data('data/ratings.dat', ['UserID', 'MovieID', 'Rating', 'Timestamp'])

    # Setup Users Dataframe
    users = load_data('data/users.dat', ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    users['Zip-code'] = users['Zip-code'].str[0]
    rating_counts = ratings['UserID'].value_counts().to_dict()
    users['RatingsCount'] = [rating_counts[i] if i in rating_counts else 0 for i in users['UserID']]

    # Setup Movies Dataframe
    movies = load_data('data/movies.dat', ['MovieID', 'Title', 'Genres'])
    movies['Genres'] = movies['Genres'].str.split('|')
    rating_counts = ratings['MovieID'].value_counts().to_dict()
    movies['RatingsCount'] = [rating_counts[i] if i in rating_counts else 0 for i in movies['MovieID']]

    # Vectorize genres and use these to cluster the movies
    binarizer = MultiLabelBinarizer()
    feature_vector = binarizer.fit_transform(movies['Genres'])
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(feature_vector)
    movies['Cluster'] = kmeans.predict(feature_vector)

    # Save the movies to a CSV file
    # movies.to_csv('movies.csv', index=False)

    # merge the ratings dataframe with the users dataframe
    user_rating_merge = pd.merge(ratings, users, on='UserID')

    # merge the merged dataframe with the movies dataframe
    user_rating_merge = pd.merge(user_rating_merge, movies, on='MovieID')

    # use the pivot function to create the matrix
    movie_ratings_matrix = user_rating_merge.pivot(index='MovieID', columns='UserID', values='Rating')

    movie_ratings_matrix = movie_ratings_matrix.fillna(0)

    model = NMF(n_components=2, init='nndsvd')
    W = model.fit_transform(movie_ratings_matrix)
    H = model.components_

    completed_matrix = pd.DataFrame(np.dot(W, H), columns=movie_ratings_matrix.columns,
                                    index=movie_ratings_matrix.index)

    user_age = input("How old are you? \n>> ").strip()
    while not user_age.isdigit():
        user_age = input("That's an invalid input. Please type in an age as a whole number.\n>> ") \
            .strip()

    user_age = int(user_age)

    if user_age < 18:
        user_age = 1
    elif user_age < 25:
        user_age = 18
    elif user_age < 35:
        user_age = 25
    elif user_age < 45:
        user_age = 35
    elif user_age < 50:
        user_age = 45
    elif user_age < 56:
        user_age = 50
    else:
        user_age = 56

    user_gender = input("What is your gender? Type in either \"M\" (male) or \"F\" (female).\n>> ").strip()
    while user_gender != 'M' and user_gender != 'F':
        user_gender = input("That's an invalid input. Please type in either \"M\" (male) or \"F\" (female).\n>> ") \
            .strip()

    zip_code = input("What's your ZIP code? \n>> ").strip()
    while len(zip_code) != 5 or not zip_code.isdigit():
        zip_code = input("That's an invalid input. Please type in a valid five-digit ZIP code.\n>> ") \
            .strip()
    zip_code = int(zip_code[0])

    try:
        occupation = int(
            input('What is your occupation? Choose from one of the following and type in the corresponding number.'
                  '\n'
                  '* 0: \"other\" or not specified \n'
                  '* 1: \"academic/educator\" \n'
                  '* 2: \"artist\" \n'
                  '* 3: \"clerical/admin\" \n'
                  '* 4: \"college/grad student\" \n'
                  '* 5: \"customer service\" \n'
                  '* 6: \"doctor/health care\" \n'
                  '* 7: \"executive/managerial\" \n'
                  '* 8: \"farmer\" \n'
                  '* 9: \"homemaker\" \n'
                  '* 10: \"K-12 student\" \n'
                  '* 11: \"lawyer\" \n'
                  '* 12: \"programmer\" \n'
                  '* 13: \"retired\" \n'
                  '* 14: \"sales/marketing\" \n'
                  '* 15: \"scientist\" \n'
                  '* 16: \"self-employed\" \n'
                  '* 17: \"technician/engineer\" \n'
                  '* 18: \"tradesman/craftsman\" \n'
                  '* 19: \"unemployed\" \n'
                  '* 20: \"writer\" \n'
                  '>> ').strip())
        if not 0 <= occupation <= 20:
            raise Exception
    except (Exception, ValueError, TypeError):
        while True:
            try:
                occupation = int(input("That's an invalid input. Please type in a valid integer "
                                       "representing an occupation from the list.\n>> ").strip())
                if 0 <= occupation <= 20:
                    break
            except (ValueError, TypeError):
                pass

    # Filter matching users
    users_filtered = users_filtered = users[(users['Gender'] == user_gender) & (users['Age'] == user_age) &
                                            (users['Occupation'] == occupation) & (users['Zip-code'] == str(zip_code))]
    while users_filtered.empty:
        if (zip_code + 1) < 10:
            zip_code += 1
        else:
            zip_code = 0

        users_filtered = users_filtered = users[(users['Gender'] == user_gender) & (users['Age'] == user_age) &
                                                (users['Occupation'] == occupation) &
                                                (users['Zip-code'] == str(zip_code))]

    predicted_user = users_filtered.loc[users_filtered['RatingsCount'].idxmax()]

    predicted_user_ratings = completed_matrix[predicted_user['UserID']]
    predicted_user_ratings = predicted_user_ratings.sort_values(ascending=False)

    recommendations = movies.iloc[pd.Index(movies['MovieID']).get_indexer(predicted_user_ratings.index)]
    recommendations = pd.concat([recommendations[recommendations['Cluster'] == 0].head(25),
                                 recommendations[recommendations['Cluster'] == 1].head(25),
                                 recommendations[recommendations['Cluster'] == 2].head(25),
                                 recommendations[recommendations['Cluster'] == 3].head(25)])

    recommendations = recommendations.sample(frac=1)

    for index, row in recommendations.iterrows():
        print('Recommendation: ' + '\033[94m' + row['Title'] + '\033[0m')
        user_command = input('Press \'n\' for the next recommendation or press \'q\' to quit. \n>> ').strip()
        while user_command.lower() != 'n' and user_command.lower() != 'q':
            user_command = input('Press \'n\' for the next recommendation or press \'q\' to quit. \n>> ').strip()
        if user_command.lower() == 'q':
            break


if __name__ == "__main__":
    main()
