
class MLModelConfigs:
    
    """ Classifier_Type: possible values: 'Binary', 'Multi' """
    cls_type   = 'Binary'  
    
    # cross-validation
    cv = 3
    
    # params for GridSearchCV
    ''' 
        @n_neighbors   -> number of neighbors to choose for computation
        @algorithm      -> algorithm used to compute nearest neighbours
        @weights        -> weight function used in the predictions
    ''' 
    knn        = { 'n_neighbors'    : [1, 3, 7, 11, 21], 
                    'algorithm'     : ('auto', 'ball_tree', 'kd_tree', 'brute'), 
                    'weights'       : ('uniform', 'distance')}

    '''
        @n_estimators    -> increasing it, increases the number of estimators which can 
                            improve the performance but also increase training time
    '''
    rf = {'n_estimators':[300, 500, 1000]} 

    '''
        @alpha -> increasing it increases regulerization in model's weight computation
    '''
    ridge = {'alpha' : [0.5, 1.75, 2.35, 10.5, 0.0005] }

    '''
        @alpha -> increasing it increases regulerization in model's weight computation
    '''
    lasso = {'alpha' : [0.01, 0.0003, 3.2, 5.12, 9.11] }

    ''' Linear Regression
        @fit_intercept  -> calculate intercept for this model, if set to True 
    '''
    lr = {'fit_intercept':[True, False]}

    rf_reg =  {
                'bootstrap': [True],
                'max_depth': [10, 20, 30],
                'max_features': ['auto'],
                'min_samples_leaf': [1, 2, 3],
                'min_samples_split': [2, 5, 10],
                'n_estimators': [300, 500, 1000]
                }

class Configs:

    '''possible values: "LabelEncoder", "OneHotEncoder" '''
    encoder = "OneHotEncoder"

    '''possible values: "StandardScaler", "MaxAbsScaler" '''
    scaler  = "StandardScaler" 

    '''Best values: 0.3, 0.2, 0.1, 0.4'''
    test_size = 0.3

    target_column = 'final_test'