import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer

# Load the movies dataset
movies = pd.read_csv('../movies.csv')

# Convert the string representation of the Genres list in each row to an actual list of genres
# 1. Remove the square brackets using strip("[]")
# 2. Remove the single quotes using replace("'", "")
# 3. Split the string into a list of genres using split(', ')
movies['Genres'] = movies['Genres'].apply(lambda x: x.strip("[]").replace("'", "").split(', '))

# Encode the Genres column using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(movies['Genres'])
genres_encoded_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

# Apply PCA to reduce the dimensions to 2
pca = PCA(n_components=2)
genres_pca = pca.fit_transform(genres_encoded_df)

# a scatter plot with color-coded clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x=genres_pca[:, 0], y=genres_pca[:, 1], hue=movies['Cluster'], palette='tab10', s=50, edgecolor='k')
plt.title('Genre-based Clustering of Movies')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Additional figures
# bar plot to show the number of movies in each cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster', data=movies, palette='tab10')
plt.title('Number of movies in each cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()

