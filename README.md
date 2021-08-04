## Name
Yee Yon Fai
yyonfai@gmail.com

# MACHINE LEARNING PIPELINE 

This project is for the prediction of Student’s O-level mathematics examination score using hyperparameter testing and machine learning alogrithm

## Directory structure
```
├── scr
|   └── configs.py // Algorithms parameters
|   └── ml_models.py // Load ML model
|   └── utils.py // Data preprocessing
├── data
    └── score.db
├── eda.ipnb // Exploratory data analysis    
├── run.py 
├── run.sh 
├── requirements.txt 



The following features are provided:

**Independent Features:**
* `number_of_siblings:`  Number of siblings
* `direct_admission:` Mode of entering the school
* `CCA:` Enrolled CCA
* `learning_style:`  Primary learning style
* `tuition:` Indication of whether the student has a tuition
* `n_male:` Number of male classmates
* `n_female:` Number of female classmates
* `gender:` Gender Type
* `age:` Age of the student
* `hours_per_week:` Number of hours student studies per week
* `attendance_rate:` Attendance rate of the student (%)
* `mode_of_transport:` Mode of transport to school
* `bag_color:` Colour of student’s bag
 
**Target Features:**
* `final_test' : Student’s O-level mathematics examination score

**Insights into Features**
1) Number of siblings seems to have an inverse relationship with the final test score. One of the reason is that they may to take care of their siblings and thus having less time to study.
2) Attendance rate, Sleeping hours and direct admission has the highest impact on student score.
3) Students with less than 7 hours of sleeping time tend to do badly in their test. 
4) Students with less than 7 hours of sleeping time tend to have lower attendance rate, they may be tired due to lack of sleep and skip school. Because they have less attendance rate, they tend to fall back on their studies and thus getting lower score.
5) Students who are have direct admission to their school tend to perform better than average students.
6) There are duplicates, missing values in the dataset and require cleaning.


| Algorithm            | Parameters                       |
| -------------------- | ---------------------------------|
| Random Forest        | n_estimators  |
| K Nearest Neighbors  | n_neighbors, algorithm, weights  |
| Lasso                | alpha                            |
| Ridge                | alpha                            |
| Linear Regression    | fit_intercept                    |


I have different range of values for parameters of these machine learning models and have used GridSearchCV technique to choose best values for final models. The values of these parameters can be changed from config.py file. In Random Forest, by increasing the value of  ‘n_estimators’ parameter increase the performance of the model, but training time also increases. 

In KNN, with n_neighbors as 15, ‘auto’ algorithm and ‘distance’ weights, the model’s performance is stable but not good as Random Forest. The performance of Random Forest is good as compared to Random Forest. Similarly, for regression algorithms only on parameter ‘alpha’ is tuned in case of Ridge & Lasso. For Linear Regression, I have tuned ‘fit_intercept’ parameter. For Ridge, the best value for alpha was around 1.75 and 0.01 for Lasso. 

For Linear Regression, the True value for ‘fit_intercept’ is best value. 

To study this problem as classification problem I have divided final test score into 2 bins assuming 0-60 as [0:’fail’] and 60-100 as [1:’pass’]. To make this problem, a multi class problem I have divided the ‘final_test’ column into 3 bins assuming 0-50 as [0:’fail’], 51-70 as [1:’average’] and 71-100 as [2:’excellent’]. The data processing pipeline includes, handling null values, standardizing data values and encoding labels for categorical columns. 

In model training Pipeline, scaling and training model is included where scaling (Standard or MinAbs) based on configs is applied on dataset before training or evaluating machine learning model. User can also provide tuning parameters for machine learning models through config file. To obtain best parameters, I have used GridSearchCV method and used best estimator for final evaluation. 

### Choice of Algorithm
Since we have a relatively large dataset, the Random forest will reduce variance part of error rather than bias part, so on a given training data set decision tree may be more accurate than a random forest. But on an unexpected validation data set, Random forest always wins in terms of accuracy.

The best random forest regressor model:
{'bootstrap': True,
 'max_depth': 20,
 'max_features': 'auto',
 'min_samples_leaf': 1,
 'min_samples_split': 10,
 'n_estimators': 500}

Mean Absolute Error (MAE): 5.474026625082996
Mean Squared Error (MSE): 54.39665613036192
Root Mean Squared Error (RMSE): 7.375408878859661
Mean Absolute Percentage Error (MAPE): 8.53
Accuracy: 91.47

## Table of Content
1) Overview of the machine learning pipeline
2) Running of the machine learning pipeline

### 1) Overview of the machine learning pipeline

#### Step a) Data-preprocessing
After the data is imported, it is preprocessed in the file using sklearn.preprocessing library

#### Step b) Splitting into train and test set
The data is then split into training set X and test set y. 

#### Step c) Encoding categorical features
After step 1, There are a large number of categorical variables such as 'gender', 'CCA', 'direct_admission', and 'learning_style' and many others that was converted into integer by using one-hot encoding and label encoder.

#### Step d) Normalization/Scaling of the data
MaxAbsScaler and StandardaScalar was used as it preserves the shape of the dataset.

#### Step e) Training on Machine learning model
Different models were trained and their R-square value is driven

Supervised Learning Regression Models used:
- Linear Regression
- Ridged Regression
- Lasso Regression
- K Nearest Neighbors Classifier

Unsupervised Learning Models
- Random Forest Classifier
- Random Forest Regressor


#### Step f) Results
The machine learning pipelin will provide you with the following results
- Model performance table (on the training set)
Table tabulating each model being trained, its performance based on scoring selected, and the best parameters that returned the scoring.

- Prediction report (on the test set)
Adj R-squared and Variance between prediction results and test set


### 2) Running of the machine learning pipeline

Machine Learning model created in with Python version 3.6.7/3.6.8 and bash script.

##### Installing Dependencies
Paste the following command on your bash terminal to download dependencies
```
pip install -r requirements.txt
```
##### Running the Machine Learning Pipeline
Paste the following command on your bash terminal to grant permission to execute the 'run.sh' file
```
chmod +x run.sh
```
Paste the following command on your bash terminal to run the machine learning programme
```
./run.sh
```




