# Clustering of clusters
This is a data classification for time series where it can return the k-nearest neighbors given two o more components (time, geography, humidity,etc).
## How does it work?
Clustering of clusters work by first choosing the components needed to evaluate the time series. For each component, all the time series in a dataset has to be sorted in regards to the component. For example, a dataset has the components for incidence,date and geography. One can first sort the time series present for different geographic places by yearly behavior (2019,2020,2021,2022,2023). Then for each sorted set of arrays, a cluster can be achieved by creating a dissimilarity matrix for every time series present in the set. When all dissimilarity matrices are made, it can then be used to create the second layer of clustering, where each dissimilarity matrix is flattened as an array (Where each value is the distance measure between one time series from another), and create a second dissimilarity matrix where the distance measure is now used to compare yearly behavior, not geographic behavior.
![Concept for cluster of clusters](./src/include/cluster_of_clusters_concept.svg)

When assembled, the nearest neighbors for a time series can be reached by searching the k-nearest years from the present year of the time series, and for each k year, the n-nearest neighbors from the time series, and so on if there are more than two layers of clustering.
For this github project a demo is used to test the viability of clustering of cluster, where the dataset is taken from the publicly available data for dengue incidence in Paraguay from 2019 to 2023: [Arbovirosis](https://dgvs.mspbs.gov.py/arbovirosis/). The dataset is tested with the following data classifications
- Clustering of clusters (this project)
- Hierarchical clustering
- Clustering (one layer)
- CART
- Random Forest
- Historical data (No data classification applied)

After processing the data, each classification then creates a forecast for the time series for the 2022 to 2023 incidence data using the following forecast models
- Naive Drift
- Auto ARIMA
- Linear Regression
- LSTM

Both CART and Random Forest are capable of forecasting data on their own, so they are excluded for the forecasting models mentioned above.
