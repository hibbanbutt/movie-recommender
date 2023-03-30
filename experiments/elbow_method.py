import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MultiLabelBinarizer

from movie_recommender import load_data


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# Setup Movies Dataframe
movies = load_data('../data/movies.dat', ['MovieID', 'Title', 'Genres'])
movies['Genres'] = movies['Genres'].str.split('|')

# Vectorize genres and use these to cluster the movies
binarizer = MultiLabelBinarizer()
feature_vector = binarizer.fit_transform(movies['Genres'])

# Use the elbow method to assess optimal k
# Source: https://predictivehacks.com/k-means-elbow-method-code-for-python/
distortions = []
K = range(1, 50)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(feature_vector)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(16, 8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.xticks(K)
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# -------------------------------------------------------------------------------------

clustering = AgglomerativeClustering(n_clusters=4, metric='cosine', linkage='ward', compute_distances=True)
plot_dendrogram(clustering, truncate_mode="level", p=3)

