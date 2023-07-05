import matplotlib.pyplot as plt
import pandas as pd
import sklearn.cluster as skl
from yellowbrick.cluster import KElbowVisualizer, silhouette_visualizer

# Load data from the CSV file
dataframe = pd.read_csv(r'countries.csv')

# Get the features of interest
X = dataframe[['Life Expectancy', 'GDP Per Capita', 'CO2 Emissions Per Capita']]

for k in range(2, 11):
    silhouette_visualizer(skl.KMeans(k, random_state=42), X, colors='yellowbrick')

# Elbow method
model = skl.KMeans()
visualizer = KElbowVisualizer(model, k=(1, 11))
visualizer.fit(X)
visualizer.show()

plt.show()

k = int(input("Enter the number of clusters k: "))

# Initialize the clustering algorithm with the user-specified parameter K
kmeansModel = skl.KMeans(n_clusters=k)

# Fit the data
kmeansModel.fit(X)

# Get the centroids
centroids = kmeansModel.cluster_centers_

# Get a list of data labels
labels = kmeansModel.predict(X)

# Add a classification label column to the dataframe
dataframe['label'] = labels

# Visualization of 2 clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Life Expectancy')
ax.set_ylabel('GDP Per Capita')
ax.set_zlabel('CO2 Emissions Per Capita')

ax.scatter(dataframe.loc[dataframe['label'] == 0, 'Life Expectancy'],
           dataframe.loc[dataframe['label'] == 0, 'GDP Per Capita'],
           dataframe.loc[dataframe['label'] == 0, 'CO2 Emissions Per Capita'],
           c="blue")
ax.scatter(dataframe.loc[dataframe['label'] == 1, 'Life Expectancy'],
           dataframe.loc[dataframe['label'] == 1, 'GDP Per Capita'],
           dataframe.loc[dataframe['label'] == 1, 'CO2 Emissions Per Capita'],
           c="hotpink")

plt.show()

