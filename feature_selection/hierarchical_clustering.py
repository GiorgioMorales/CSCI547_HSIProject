import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


class hierarchical_clustering:

    def __init__(self, data, produce, num_waves):
        self.data = np.array(data)
        self.clusters = None
        self.link = None
        self.produce = produce
        self.num_waves = num_waves

    def cluster(self):
        '''
        Use hierarchical clustering to cluster
        the data, and return the clusters
        calculated in this manner
        '''

        # method='centroid' and metric='euclidean' has the
        # lowest cophenetic correlation for the three class
        # banana dataset
        self.max_d = 60
        self.link = linkage(self.data, method='centroid', metric='euclidean')
        self.clusters = fcluster(self.link, self.max_d, criterion='distance')
        return self.clusters

    def cophenetic_corr(self):
        """
        Display the cophenetic correlation of
        the dataset
        """

        c, coph_dists = cophenet(self.link, pdist(self.data))
        print (c)

    def display_dendrogram(self):
        """
        Plot the resulting dengrogram from the
        hierarchical clustering
        """

        # See: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram - ' + self.produce + ' - ' + str(self.num_waves))
        plt.axhline(self.max_d)
        plt.xlabel('Index')
        plt.ylabel('Distance (Euclidean)')
        dendrogram(
            self.link,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
        )
        plt.show()

    def pca_plot(self):
        """
        Display the 3-d plot of the first three
        principal components of the data. This can
        be a rough way to figure out if the
        clustering algorithm produced the correct
        number of clusters (compare to dendrogram)
        color the points according to their cluster
        assignment
        """

        pca = PCA(n_components=3)
        pca.fit(self.data)
        new_data = pca.transform(self.data)
        fig = plt.figure()
        # fig.suptitle('Hierarchical Cluster Assignment - ' + self.produce + ' - ' + str(self.num_waves))
        fig.suptitle('Hierarchical Subpopulation Clustering', size=50)
        ax = fig.add_subplot(111, projection='3d')
        plt.tick_params(labelsize=32)
        ax.set_xlabel('PC1', size=40, labelpad=43)
        ax.set_ylabel('PC2', size=40, labelpad=43)
        ax.set_zlabel('PC3', size=40, labelpad=43)
        ax.scatter(new_data[:, 0], new_data[:, 1], new_data[:, 2], c=self.clusters)
        plt.show()
