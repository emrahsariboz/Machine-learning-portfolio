# Regression Analyses

## The general structure of the command line argument is as follows:

### python [file_name] [regression_name] [--alpha] [--solver] [--degree] [dataset_name]

Double hyphen indicates optional arguemtns

### For Linear Regression Analysis

    python main.py linear_reg BostonHousing.csv
    python main.py linear_reg cali_renaw_cleaned.csv
    
### For normal equation

    python main.py normal_reg BostonHousing.csv
    
### For Ridge Regression analysis

    python main.py ridge_reg BostonHousing.csv

    python main.py ridge_reg cali_renaw_cleaned.csv
    
    ### For a specific alpha and/or solver 
    
    python main.py ridge_reg --alpha 2 --solver svd BostonHousing.csv

### For Lasso Regression analysis

    python main.py lasso_reg BostonHousing.csv
    
    python main.py lasso_reg cali_renaw_cleaned.csv
    
    ### For a specific degree
    
    python main.py lasso_reg --degree 2 cali_renaw_cleaned.csv

### For Ransac Regression analysis
	
    python main.py ransac_reg BostonHousing.csv

    python main.py ransac_reg cali_renaw_cleaned.csv
	
### Polynomial Featured Linear Regression

    python main.py poly_lr BostonHousing.csv
	
    python main.py ransac_reg cali_renaw_cleaned.csv
    
    ### For a speficit degree
    
    python main.py poly_lr --degree 3 BostonHousing.csv
    
    

#####  All plots are inside the .zip folder.
##### Preprocessing details of the Cali. R. dataset can be found in dataCleaning.py 
#### In the assignment, it is asked to test normal equation on housing dataset, thus report doesnt have analysis for Cali. R. dataset.
#### However, code supports different dataset as well. (it works on cali. r. dataset as well).