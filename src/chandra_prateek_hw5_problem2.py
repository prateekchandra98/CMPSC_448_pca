
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn import decomposition
from sklearn import datasets
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')

def varianceAllK(X):
    d = dict();
    #----------------------------------1 PRINCIPAL COMPONENTS-----------------------------------
    print('1 Dimension PCA')
    pca1 = decomposition.PCA(n_components=1)
    pca1.fit(X)
    pca1.transform(X)
    print ('PCA 1 Variance: ',pca1.explained_variance_)
    print ('PCA 1 Variance Ratio', pca1.explained_variance_ratio_, '\n')
    d['PCA 1 Dimension Variance'] = pca1.explained_variance_

    
    #----------------------------------2 PRINCIPAL COMPONENTS-----------------------------------
    print('2 Dimension PCA')
    pca2 = decomposition.PCA(n_components=2)
    pca2.fit(X)
    pca2.transform(X)
    print ('PCA 2 Variance: ',pca2.explained_variance_, '\n')
    d['PCA 2 Dimension Variance'] = pca2.explained_variance_
    
    #----------------------------------3 PRINCIPAL COMPONENTS-----------------------------------
    print('3 Dimension PCA')
    pca3 = decomposition.PCA(n_components=3)
    pca3.fit(X)
    pca3.transform(X)
    print ('PCA 3 Variance: ',pca3.explained_variance_, '\n')
    d['PCA 3 Dimension Variance'] = pca3.explained_variance_
    
    #------------------------------------4 PRINCIPAL COMPONENTS------------------------------------
    print('4 Dimension PCA')
    pca4 = decomposition.PCA(n_components=4)
    pca4.fit(X)
    pca4.transform(X)
    print ('PCA 4 Variance: ',pca4.explained_variance_, '\n')
    d['PCA 4 Dimension Variance'] = pca4.explained_variance_
    
    return d
    
def graph(X, y, title, target_names, supervised):
    cluster_name = ['Cluster 0','Cluster 1','Cluster 2']
    pca2 = decomposition.PCA(n_components=2)
    twoPrincipalComponents = pca2.fit_transform(X)
    plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
        if supervised:
            plt.scatter(twoPrincipalComponents[y==i,0], twoPrincipalComponents[y==i,1], c=c, label=target_names[i])
        else:
            plt.scatter(twoPrincipalComponents[y==i,0], twoPrincipalComponents[y==i,1], c=c, label=cluster_name[i])
    plt.legend()
    plt.title(title)
    
    plt.show()

def kmeansGraph(X, y, title, target_names):
    kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 50)
    y_kmeans = kmeans.fit_predict(X)
    graph (X,y_kmeans, title, target_names, False)
    
if __name__ == '__main__':

    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = list(iris.target_names)
    
    mean = np.mean(X, axis=0)    
    
    standardDev = np.std(X, axis=0)
    
    print('-----------------Raw Data----------------\n')
    d = varianceAllK(X)
    
    print('--------Centering or Mean Removal--------\n')
    center_scaler=StandardScaler(copy=True,with_mean=True, with_std=False)
    centered_X = center_scaler.fit_transform(X)
    dcenter = varianceAllK(centered_X)
    
    print('------Scale Features in range [0,1]------\n')
    min_max_scaler=MinMaxScaler()
    min_max_X = min_max_scaler.fit_transform(X)
    dscale = varianceAllK(min_max_X)
    
    print('-------------Standardization-------------\n')
    std_scaler=StandardScaler(copy=True,with_mean=True, with_std=True)
    std_X = std_scaler.fit_transform(X)
    dstd = varianceAllK(std_X)
    
    print('--------------Normalization--------------\n')
    norm_X = normalize(X, norm='l2')
    dnorm = varianceAllK(norm_X)
    
    print('-----------------------PCA 2 DIMENSION----------------------')
    graph (X,y, 'PCA for raw dataset', target_names, True)
    graph (centered_X,y,'PCA for centered dataset', target_names, True)
    graph (min_max_X,y, 'PCA for scaled dataset', target_names, True)
    graph (std_X,y, 'PCA for standardized dataset', target_names, True)
    graph (norm_X,y, 'PCA for normalized dataset', target_names, True)
    
    print('------------------------- K Means++ ------------------------')
    kmeansGraph (X,y, 'K Means++ for raw dataset', target_names)
    kmeansGraph (centered_X,y,'K Means++ for centered dataset', target_names)
    kmeansGraph (min_max_X,y, 'K Means++ for scaled dataset', target_names)
    kmeansGraph (std_X,y, 'K Means++ for standardized dataset', target_names)
    kmeansGraph (norm_X,y, 'K Means++ for normalized dataset', target_names)
    
    
    


    
    
    
    