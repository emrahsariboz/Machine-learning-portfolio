# Clustering Analysis

## The general structure of the command line argument is as follows:

### python [file_name] [clustering_alo] [--n_cluster] [--criterion] [--epsilon] [--min_samples] [dataset_name]

Double hyphen indicates optional arguemtns

### For K-means analysis

    python main.py k_means iris.csv
    python main.py k_means mnist
    
    ### For a specific hyperparameters
    
    python main.py k_means --n_cluster 3 iris.csv
    

### For Agglomerative Clustering - SciPy

    python main.py scipy_agg iris.csv
        
    python main.py scipy_agg mnist
    
    ### For a specific hyperparameters
    
    python main.py scipy_agg --n_cluster 3 --criterion complete iris.csv
    
### For Agglomerative Clustering - sklearn

    python main.py sklearn_agg iris.csv
	    
    python main.py sklearn_agg mnist
    
    ### For a specific hyperparameters
    
    python main.py sklearn_agg --n_cluster 3 --criterion single iris.csv
### For DBSCAN analysis
	
    python main.py dbscan iris.csv

    python main.py dbscan mnist	 
    ### For a specific hyperparameters
    
    python main.py dbscan --epsilon 0.1 --min_sample 5 iris.csv
    
    
#####  All plots are inside the .zip folder.
#### Please note that when you use mnist dataset for the DBSCAN algorithm, it will take long time to find optimal epsilon and minimum sample.