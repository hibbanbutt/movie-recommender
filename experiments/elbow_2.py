import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


movies_df = pd.read_parquet("C:/Users/Ahmed/Documents/GitHub/movie-recommender/movies_processed.parquet")


#Just in case: Drop any rows with missing intro_vector values:
movies_df = movies_df.dropna(subset=["intro_vector"])

#Create a NumPy array of intro_vectors for clustering:
intro_vectors = np.stack(movies_df["intro_vector"].values)

# Determine the optimal number of clusters using the elbow method
inertia_values = []
num_clusters = range(1, 200)

for k in num_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(intro_vectors)
    inertia_values.append(kmeans.inertia_)

plt.plot(num_clusters, inertia_values, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

