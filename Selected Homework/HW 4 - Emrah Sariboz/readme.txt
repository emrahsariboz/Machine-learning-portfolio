# Ensemble approaches

## The general structure of the command line argument is as follows:

### python [file_name] [classifier_name] [dataset_name]


    python main.py randomForest digit
    python main.py randomForest mammographic_masses_converted_cleaned.csv
    
    python main.py bagging digit
    python main.py bagging mammographic_masses_converted_cleaned.csv
    
    python main.py adaboost digit
    python main.py adaboost mammographic_masses_converted_cleaned.csv


### I imputed missing values of the MM dataset with median value of the corresponding value.
### You can find the detailed pre-processing inside the dataPreprocessing.py
### Inside the folder, you can find raw dataset as well as the final version (preprocessed) [final version of the dataset is: mammographic_masses_converted_cleaned.csv]
