# Ensemble approaches

## The general structure of the command line argument is as follows:

### python [file_name] [technique] [component] [max_depth] [min_split] [--kernel] [--solver] [--shrinkage] [dataset_name]

Double hyphen indicates optional arguemtns

### For PCA analysis

    python main.py pca 140 9 5 mnist
    python main.py pca 2 5 3 iris.csv
    
### For kPCA analysis

    python main.py kernel_pca 140 9 5 --kernel poly mnist
    python main.py kernel_pca 2 5 3 --kernel poly iris.csv
    
### For LDA analysis

    python main.py lda 140 9 5 --solver svd  mnist
    or
    python main.py lda 140 9 5 --solver eigen --shrinkage auto mnist
    
    
    python main.py lda 2 5 3 --solver svd iris.csv
    or
    python main.py lda 2 5 3 --solver eigen --shrinkage auto iris.csv
    

#####  For LDA analysis, I wrote two different command line just in case if you want to use eigenvalue decomposition technique which allows you to specify shrinkage as well.
##### Iris dataset is in the folder as well. [iris.csv]
