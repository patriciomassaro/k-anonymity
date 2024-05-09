import numpy as np

from utils.anonymization.anonymize import Anonymizer
from utils.Preprocessing import preprocess_data
from utils.Optimizer import Optimizer

import pandas as pd
import logging
import json
import os
import glob


# create a function that returns the target given the dataset
def get_target(dataset:str):
    if dataset == 'adult':
        return 'salary-class'
    elif dataset == 'cahousing':
        return 'median_house_value'
    elif dataset == 'cmc':
        return 'method'
    else:
        raise ValueError('Dataset not found')

def instanciate_log(logfilename:str = 'app.log'):
    # Remove the log file if it exists
    try:
        os.remove(logfilename)
    except OSError:
        pass
    # Instanciate a logging with debug level
    logging.basicConfig(filename=logfilename, filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)


DATASET_SIZE_DICT = {
    'adult': 32561,
    'cahousing': 20640,
    'cmc': 1473
}

if __name__ == '__main__':
    """
    This is the main function of the script 
    """
    #create a string with the date and time
    date_time = pd.to_datetime('today').strftime('%Y-%m-%d_%H-%M-%S')
    METRICS_JSON_PATH = f"{date_time}_metrics.json"
    # create a random number for the seed
    seed = np.random.randint(0,1000)
    
    # Instanciate the log
    instanciate_log()


    datasets = [
        "cahousing",
        "adult",
        # "cmc"
    ]

    methods = ["mondrian",
               "classic_mondrian",
               "datafly"
               ]
    # Create a list of numbers from 2 to 100 with 20 steps using numpy
    ks= list(np.floor(np.linspace(2,100,20)))

    # Anonymization
    """
    Path to the anonymized datasets:  Results/%DATASET%/%METHOD%/%DATASET%_anonymized_%K%.csv
    """
    # for dataset in datasets:
    #     for method in methods:
    #         for k in ks:
    #
    #             file_name = dataset+"_anonymized_"+str(k)+".csv"
    #
    #             if os.path.exists("results/"+dataset+"/"+method+"/"+file_name):
    #                 logging.info(file_name + " already exists!")
    #             else:
    #                 logging.info("Creating " + file_name)
    #                 args = {'method': method, 'k': k, 'dataset': dataset}
    #                 anonymizer = Anonymizer(args)
    #                 anonymizer.anonymize()




    # Preprocessing loop
    """
    We take the anonymized datasets, apply preprocessing functions. Result : A folder with datasets ready to be used by the optimizer

    Path to the anonymized preprocessed datasets:  Results/%DATASET%/%METHOD%/%DATASET%_anonymized_%K%_prep.csv
    """

    # Get all the path of the anonymized datasets using blob
    anonymized_datasets =  [fn for fn in glob.glob("results/**/**/*_anonymized_*.csv") if not 'prep' in fn]

    for anonymized_dataset_path in anonymized_datasets:
        preprocessed_dataset_path = anonymized_dataset_path.replace('.csv','_prep.csv')
        logging.info(f"Preprocessing {anonymized_dataset_path}")
        # read the dataset and preprocess it
        if os.path.exists(preprocessed_dataset_path):
                    logging.info( preprocessed_dataset_path + " already exists!")
        else:
            logging.info("Creating " + preprocessed_dataset_path)
            data = pd.read_csv(anonymized_dataset_path, sep=';', header=0, encoding='ascii')
            preprocessed_dataset = preprocess_data(data)
            # Save the preprocessed datasets in the same folder
            preprocessed_dataset.to_csv(preprocessed_dataset_path, index=False, sep=';')
            logging.info(f"Saving to {preprocessed_dataset_path}")

    ml_algorithms = ['xgboost','randomforest']

    # Dictionary to save the metrics
    metrics = {
        'dataset':[],
        'method':[],
        'k':[],
        'ml_algorithm':[],
        'metrics':[],
        'dataset_size':[]
    }

    datasets_folders = [fn for fn in glob.glob("results/*")]
    for dataset_path in datasets_folders:
        dataset_name = dataset_path.split('/')[-1]
        # get the target given the dataset
        target = get_target(dataset_name)
        methods_folders = [fn for fn in glob.glob(dataset_path+"/*")]

        for method_path in methods_folders:
            method_name = method_path.split('/')[-1]
            preprocessed_datasets =  [fn for fn in glob.glob(method_path+"/*_anonymized_*.csv") if 'prep' in fn]

            for preprocessed_dataset in preprocessed_datasets:
                k = preprocessed_dataset.split('_')[-2]
                data = pd.read_csv(preprocessed_dataset, sep=';', header=0, encoding='ascii')

                for ml_algorithm in ml_algorithms:
                    logging.info(f"Optimizing {dataset_name} with {method_name} and {k} for {ml_algorithm}")
                    opt = Optimizer(model_type=ml_algorithm,
                                    data=data,
                                    target=target,
                                    cv_splits=4,
                                    max_evals=20,
                                    seed=seed)
                    opt.optimize()
                    logging.info(f" Best Parameters: {opt.best_parameters}")
                    best_model = opt.train_best_model()
                    # make predictions from the best model in the test set
                    opt.make_predictions_from_best_model()
                    # save info to the metrics dictionary
                    metrics['dataset'].append(dataset_name)
                    metrics['method'].append(method_name)
                    metrics['k'].append(k)
                    metrics['ml_algorithm'].append(ml_algorithm)
                    metrics['metrics'].append(opt.report_metrics())
                    metrics['dataset_size'].append(DATASET_SIZE_DICT[dataset_name])



                logging.info(f" Best Parameters: {opt.best_parameters}")
                # train the best model

    with open(f'metrics/{METRICS_JSON_PATH}', 'w') as f:
        json.dump(metrics, f)





