# # Import the relevant Scikit-learn functions and Classes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from src.configs import Configs, MLModelConfigs

# Machine Learning Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression, Lasso

# Model Evaluation Metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

class Classifiers():
    
    @classmethod
    def get_scaler(cls):
        ''' return scaler function based on config file
        '''

        if Configs.scaler == 'StandardScaler':
            scaler = ('scaler', StandardScaler())
        
        elif Configs.scaler == 'MaxAbsScaler':
            scaler = ('scaler', MaxAbsScaler())
        
        else:
            raise Exception(f"Invalid scaler provided -> {Configs.scaler}")
        
        return scaler

    @classmethod
    def build_models(cls):
        print('building classifiers...')

        scaler = cls.get_scaler()
        knn = GridSearchCV(estimator=KNeighborsClassifier(), 
                                param_grid=MLModelConfigs.knn, cv=MLModelConfigs.cv)
        
        rf  = GridSearchCV(estimator=RandomForestClassifier(), 
                                param_grid=MLModelConfigs.rf, cv=MLModelConfigs.cv)
        
        knn_pipe= Pipeline([scaler, ('KNN', knn)])
        rf_pipe = Pipeline([scaler, ('Random Forest', rf)]) 
        return [{'name':'KNN', 'model':knn_pipe}, {'name':'Random Forest', 'model':rf_pipe}]

    @classmethod
    def fit(cls, train_X, train_Y):
        models = cls.build_models()

        print('fitting classifiers...')
        for model_config in models:
            print(f"fitting '{model_config['name']}'...")
            model_config['model'].fit(train_X, train_Y)

        return models 

    @classmethod
    def evaluate(cls, test_X, test_Y, trained_models):
        
        print('evaluating models...')
        for model_config in trained_models:
            model_name = model_config['name']
            print(f"evaluating '{model_name}'")

            grid = model_config['model'].named_steps[model_name]
            print('best_parameters: ', grid.best_params_)
            score = round(model_config['model'].score(test_X, test_Y), 2)
            print(f"'{model_config['name']}' score: {score}")
    
    @classmethod
    def run(cls, x_train, y_train, x_test, y_test):
        models = cls.fit(x_train, y_train)
        cls.evaluate(x_test, y_test, models)


class Regressors():
    
    @classmethod
    def get_scaler(cls):
        ''' return scaler function based on config file
        '''
        
        if Configs.scaler == 'StandardScaler':
            scaler = ('scaler', StandardScaler(with_mean=False))
        
        elif Configs.scaler == 'MaxAbsScaler':
            scaler = ('scaler', MaxAbsScaler())
        
        else:
            raise Exception(f"Invalid scaler provided -> {Configs.scaler}")
        
        return scaler

    @classmethod
    def build_models(cls):
        print('building regressors...')
         
        scaler = cls.get_scaler()
        _ridge  = GridSearchCV(estimator=Ridge(), param_grid=MLModelConfigs.ridge, 
                                                            cv=MLModelConfigs.cv )
        lr      = GridSearchCV(estimator=LinearRegression(), param_grid=MLModelConfigs.lr, 
                                                                    cv=MLModelConfigs.cv )
        _lasso  = GridSearchCV(estimator=Lasso(), param_grid=MLModelConfigs.lasso, 
                                                            cv=MLModelConfigs.cv )
        rf      = GridSearchCV(estimator = RandomForestRegressor(random_state = 42), 
                    param_grid = MLModelConfigs.rf_reg, cv = MLModelConfigs.cv, verbose = 2)  


        return [{   'name'  : 'Ridge', 
                    'model' : Pipeline([scaler, ('Ridge', _ridge)] )},
                {   'name'  : 'Lasso', 
                    'model' : Pipeline([scaler, ('Lasso', _lasso)])}, 
                {   'name'  : 'Linear Regression', 
                    'model' : Pipeline([scaler, ('Linear Regression', lr)])},
                {   'name'  : 'Random Forest Regressor', 
                    'model' : Pipeline([scaler, ('Random Forest Regressor', rf)])
                }]
                

    @classmethod
    def fit(cls, train_X, train_Y):
        models = cls.build_models()

        print('fitting regressors...')
        for model_config in models:
            print(f"fitting '{model_config['name']}'...")
            model_config['model'].fit(train_X, train_Y)

        return models 

    @classmethod
    def evaluate(cls, test_X, test_Y, trained_models):
        
        print('evaluating models...')
        for model_config in trained_models:
            model_name = model_config['name']
            print(f"evaluating '{model_name}'")

            grid = model_config['model'].named_steps[model_name]
            
            print('best_parameters: ', grid.best_params_)
            pred_Y = model_config['model'].predict(test_X) 
        
            r_squared   = round(r2_score(test_Y, pred_Y), 2)
            rmse        = round(mean_squared_error(test_Y, pred_Y), 2)
            print(f'r_sequared error: {r_squared}')
            print(f'RMSE: {rmse}')

    @classmethod
    def run(cls, x_train, y_train, x_test, y_test):
        models = cls.fit(x_train, y_train)
        cls.evaluate(x_test, y_test, models)




