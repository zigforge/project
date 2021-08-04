from src.ml_models import Classifiers, Regressors
from src.utils import DataReader, DataProcessor

# read data from database table
df = DataReader.read_db(db_file='data/score.db', table='score', _type='df')

x_train, y_train, x_test, y_test = DataProcessor.process(df.copy(), for_classifier=True)
Classifiers.run(x_train, y_train, x_test, y_test)

x_train, y_train, x_test, y_test = DataProcessor.process(df.copy())
Regressors.run(x_train, y_train, x_test, y_test)