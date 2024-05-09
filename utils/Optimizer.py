from hyperopt import hp,Trials,STATUS_OK,fmin,tpe
from hyperopt.pyll import scope

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,roc_auc_score,mean_squared_error,mean_absolute_error,classification_report

import xgboost as xgb
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)
XGBOOST_SEARCH_SPACE = {
            'learning_rate': hp.loguniform('learning_rate', -4, -1),
            'max_depth': scope.int(hp.uniform('max_depth', 3, 10)),
            'min_child_weight': hp.loguniform('min_child_weight', -2, 2),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'gamma': hp.loguniform('gamma', -2, 2),
            'alpha': hp.loguniform('alpha', -2, 2),
            'lambda': hp.loguniform('lambda', -2, 2),
        }

RF_SEARCH_SPACE = {
            'n_estimators': scope.int(hp.uniform('n_estimators', 100, 1000)),
            'max_depth': scope.int(hp.uniform('max_depth', 3, 10)),
            'min_samples_split': scope.int(hp.uniform('min_samples_split', 2, 10)),
            'min_samples_leaf': scope.int(hp.uniform('min_samples_leaf', 1, 10)),
            'n_jobs': -1,
}

class Optimizer:
    """
    This class is used to optimize the hyperparameters of a model using hyperopt

    Parameters
    ----------
    model_type: str ( 'xgboost' or 'randomforest')
        The model type to optimize
    data: pd.DataFrame
        The data to use for optimization, it should be already preprocessed
    seed: int
        The seed to use for reproducibility
    target: str
        The target column name
    max_evals: int
        The maximum number of evaluations to perform by hyperopt
    cv_splits: int
        The number of cross validation splits to perform
    """
    def __init__(self, model_type:str,data:pd.DataFrame,seed:int, target:str, max_evals:int=10, cv_splits:int=5):
        self.model = model_type
        self.max_evals = max_evals
        self.trials = Trials()
        self.best = None
        self.splits = cv_splits
        self.seed = seed
        self.target = target
        self.data = data


    def define_problem_type(self):
        """
        This function is used to define the problem type based on the unique values of the target
        """
        if len(self.train_target.unique()) > 5:
            logger.info("More than 5 unique values in the target, so it is a regression classification problem")
            self.problem_type = 'regression'
        elif len(self.train_target.unique()) == 2:
            logger.info("2 unique values in the target, so it is a binary classification problem")
            self.problem_type = 'binary'
        else:
            logger.info("More than 2 unique values in the target but less than 5, so it is a multiclass classification problem")
            self.problem_type = 'multiclass'


    def split_into_train_and_test(self):
        logger.info("Splitting the data into train and test")
        train,test = train_test_split(self.data,test_size=0.2,random_state=self.seed)
        self.train_features = (train.drop(self.target,axis=1))
        self.train_target = train[self.target]
        self.test_features = (test.drop(self.target,axis=1))
        self.test_target = test[self.target]

    def define_search_space(self):
        """
        This function is used to define the search space based on the model type
        """
        if self.model == 'xgboost':
            self.search_space = self.complete_search_space(XGBOOST_SEARCH_SPACE)
        elif self.model == 'randomforest':
            self.search_space = self.complete_search_space(RF_SEARCH_SPACE)

    def complete_search_space(self,search_space:dict):
        """
        takes the search space and completes it based on the model type and problem type
        """
        # add the objective function to the search space based on the model type and problem type
        if self.model == 'xgboost':
            logger.info("Using XGBoost as the model")
            if self.problem_type == 'regression':
                search_space['objective'] = 'reg:squarederror'
                search_space['eval_metric'] = 'rmse'
                self.metric = 'rmse'
            elif self.problem_type == 'binary':
                search_space['objective'] = 'binary:logistic'
                search_space['eval_metric'] = 'logloss'
                self.metric='logloss'
            elif self.problem_type == 'multiclass':
                search_space['objective'] = 'multi:softmax'
                search_space['eval_metric'] = 'mlogloss'
                search_space['num_class'] = len(self.train_target.unique())
                self.metric='mlogloss'

        return search_space

    def function_to_optimize(self, params):
        """
        This function is used to optimize the search space based on the model type and problem type,
        In xgb, the search space already considers the problem type
        In RF, we have to add the objective function based on the problem type
        """
        if self.model == 'xgboost':
            # convert data and target to DMatrix
            data = xgb.DMatrix(self.train_features, self.train_target)
            # perform a cross validation with the given parameters and return the  mean evaluation metric 
            cv_results = xgb.cv(params, data, nfold=self.splits,num_boost_round=100)
            logger.info(f"CV results: {cv_results[f'test-{self.metric}-mean'].iloc[-1]}")
            return {'status':STATUS_OK, 'loss':cv_results[f'test-{self.metric}-mean'].iloc[-1], 'attributes':params}
        elif self.model == 'randomforest':
            # perform a cross validation with the given parameters and return the  mean evaluation metric 
            if self.problem_type == 'regression':
                cv_results = cross_val_score(RandomForestRegressor(**params), self.train_features, self.train_target, cv=self.splits,error_score='raise',scoring="neg_root_mean_squared_error")
                # We want to minimize the negative mean squared error, so we multiply by -1
                cv_results = -cv_results
                logger.info(f"CV results: {cv_results}")
            elif self.problem_type == 'binary':
                cv_results = cross_val_score(RandomForestClassifier(**params), self.train_features, self.train_target, cv=self.splits,error_score='raise',scoring="neg_log_loss")
                # We want to minimize the neg log loss, so we multiply by -1
                cv_results = -cv_results
                logger.info(f"CV results: {cv_results}")
            elif self.problem_type == 'multiclass':
                cv_results = cross_val_score(RandomForestClassifier(**params), self.train_features, self.train_target, cv=self.splits,error_score='raise',scoring="f1_macro")
                # We want to maximize the F1 but fmin only minimizes, so we multiply by -1
                cv_results = -cv_results
                logger.info(f"CV results: {cv_results}")
                
            return {'status':STATUS_OK, 'loss':np.mean(cv_results), 'attributes':params}

            
            
    def optimize(self):
        """
         Perform the optimization of the search space using hyperopt fmin function
        """
        #split the data into train and test
        self.split_into_train_and_test()
        # determine the problem type
        self.define_problem_type()
        self.define_search_space()
        # Log everyting
        logger.info(f"Starting the optimization")
        logger.info(f"model : {self.model}")
        logger.info(f"problem type : {self.problem_type}")
        logger.info(f"target : {self.target}")
        logger.info(f"problem type : {self.problem_type}")


        # optimize the search space
        self.best_parameters = fmin(self.function_to_optimize, self.search_space, algo=tpe.suggest,
                         max_evals=self.max_evals, trials=self.trials)
        
        # return the best parameters
        return self.best_parameters

    
    def train_best_model(self):
        """
        This function is used to train the best model found during the optimization
        """
        logger.info("Training the best model found during the optimization")
        self.best_parameters = self.complete_search_space(self.best_parameters)
        # train the model with the best parameters found during the optimization
        if self.model == 'xgboost':
            # convert parameters to integers
            self.best_parameters['max_depth'] = int(self.best_parameters['max_depth'])
            # Train the model            
            self.best_model = xgb.train(self.best_parameters, xgb.DMatrix(self.train_features, self.train_target),num_boost_round=100)
        elif self.model == 'randomforest':
            # Convert int parameters to integers
            self.best_parameters['n_estimators'] = int(self.best_parameters['n_estimators'])
            self.best_parameters['max_depth'] = int(self.best_parameters['max_depth'])
            self.best_parameters['min_samples_split'] = int(self.best_parameters['min_samples_split'])
            self.best_parameters['min_samples_leaf'] = int(self.best_parameters['min_samples_leaf'])
            
            # Set regressor or classifier
            if (self.problem_type == 'binary') or (self.problem_type == 'multiclass'):
                self.best_model = RandomForestClassifier(**self.best_parameters)
            else:
                self.best_model = RandomForestRegressor(**self.best_parameters)
            
            # Train the model
            self.best_model.fit(self.train_features, self.train_target)            
        
        # return the best model
        return self.best_model

    def make_predictions_from_best_model(self):
        """
        This function is used to check the performance of the best model found during the optimization on the test set
        """
        logger.info("Making predictions from the best model found during the optimization")
        # predict labels and probabilities in case the problem is binary
        if self.problem_type == 'binary':
            if self.model == 'xgboost':
                self.test_proba_predictions = self.best_model.predict(xgb.DMatrix(self.test_features))
                self.test_label_predictions = np.where(self.test_proba_predictions > 0.5, 1, 0)
            elif self.model == 'randomforest':
                self.test_proba_predictions = self.best_model.predict_proba(self.test_features)[:,1]
                self.test_label_predictions = self.best_model.predict(self.test_features)
        # predict labels in case the problem is regression
        elif self.problem_type == 'regression':
            if self.model == 'xgboost':
                self.test_reg_predictions = self.best_model.predict(xgb.DMatrix(self.test_features))
            elif self.model == 'randomforest':
                self.test_reg_predictions = self.best_model.predict(self.test_features)
        elif self.problem_type == 'multiclass':
            if self.model == 'xgboost':
                self.test_label_predictions = self.best_model.predict(xgb.DMatrix(self.test_features))
            elif self.model == 'randomforest':
                self.test_label_predictions = self.best_model.predict(self.test_features)


    def report_metrics(self):
        """
        This function is used to report the performance of the best model found during the optimization on the test set
        """
        logger.info("Reporting the performance of the best model found during the optimization on the test set")
        # report the performance of the best model found during the optimization on the test set
        metrics_dict = {}
        if self.problem_type == 'binary':
            # REPORT
            logger.info(f"classification report: {classification_report(self.test_target, self.test_label_predictions)}")
            # save it to the metrics dictionary
            metrics_dict['classification_report'] = classification_report(self.test_target, self.test_label_predictions,output_dict=True)

            # ROC
            logger.info(f"ROCAUC: {roc_auc_score(self.test_target, self.test_proba_predictions)}")
            # save it to the metrics dictionary
            metrics_dict['ROCAUC'] = roc_auc_score(self.test_target, self.test_proba_predictions)
        elif self.problem_type == 'regression':
            logger.info(f"RMSE: {np.sqrt(mean_squared_error(self.test_target, self.test_reg_predictions))}")
            # save it to the metrics dictionary
            metrics_dict['RMSE'] = np.sqrt(mean_squared_error(self.test_target, self.test_reg_predictions))
        elif self.problem_type == 'multiclass':
            metrics_dict['classification_report'] = classification_report(self.test_target, self.test_label_predictions,output_dict=True)
            # add the micro f1 score
            metrics_dict['micro_f1'] = f1_score(self.test_target, self.test_label_predictions, average='micro')
            # add the macro f1 score
            metrics_dict['macro_f1'] = f1_score(self.test_target, self.test_label_predictions, average='macro')

        return metrics_dict


        

        





