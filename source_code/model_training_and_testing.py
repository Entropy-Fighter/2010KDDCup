import numpy as np
import pandas as pd
import copy
from sklearn.neural_network import MLPRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBClassifier
import lightgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

training_set = pd.read_csv("./train_tmp.csv", sep = "\t")
training_set = training_set.dropna()
testing_set = pd.read_csv("./test_tmp.csv", sep = "\t")
testing_set = testing_set.dropna()

training_Y = np.array(training_set["Correct First Attempt"]).astype(float).ravel()
training_X = training_set.drop(["Correct First Attempt"], axis = 1)

testing_Y = np.array(testing_set['Correct First Attempt']).astype(float).ravel()
testing_X = testing_set.drop(['Correct First Attempt'], axis = 1)


# Root Mean Squared Error (RMSE)
def RMSE(arr1, arr2):
    return np.sqrt(np.mean((arr1 - arr2) ** 2))

def recordBestPrediction():
    model = RandomForestRegressor(n_estimators = 190, max_depth = 15, max_leaf_nodes = 500)
    model.fit(training_X, training_Y)
    testing_set_has_nan = pd.read_csv("./test_tmp.csv", sep = "\t")
    tmp_X = testing_set_has_nan.drop(['Correct First Attempt'], axis = 1)
    tmp_Y = np.array(testing_set_has_nan['Correct First Attempt']).astype(float).ravel()
    prediction = model.predict(tmp_X)
    prediction_no_nan = model.predict(testing_X)
    print("Best RMSE = ", RMSE(testing_Y, prediction_no_nan))
    # for i, item in enumerate(prediction_no_nan):
    #     if item >= 0.5:
    #         prediction_no_nan[i] = 1.0
    #     else:
    #         prediction_no_nan[i] = 0.0
    # print("Accuracy = ", 1 - np.mean(np.abs(prediction_no_nan - testing_Y)))

    for i, y in enumerate(tmp_Y):
        if np.isnan(y):
            tmp_Y[i] = prediction[i]
            # if prediction[i] >= 0.5:
            #     tmp_Y[i] = 1.0
            # else:
            #     tmp_Y[i] = 0.0
    testing_set_incomplete = pd.read_csv("../data/test.csv", sep = "\t")
    testing_set_incomplete["Correct First Attempt"] = tmp_Y
    testing_set_incomplete.to_csv("../data/test_output.csv", sep = "\t", index = False)
    print("Prediction has been written to test_output.csv")

recordBestPrediction()

def decisionTree(x, y, tx, ty):
    model = tree.DecisionTreeClassifier()
    model.fit(x, y)
    py = model.predict(tx)
    print("DecisionTree RMSE = ", RMSE(ty, py))

def lightGBMClassify(x, y, tx, ty):
    model = lightgbm.LGBMClassifier()
    model.fit(x, y)
    py = model.predict(tx)
    print("LightGBM RMSE = ", RMSE(ty, py))

def XGBoostClassify(x, y, tx, ty):
    model = XGBClassifier()
    model.fit(x, y)
    py = model.predict(tx)
    print("XGBoost RMSE = ", RMSE(ty, py))
    
def randomForestClassify(x, y, tx, ty):
    model = RandomForestClassifier()
    model.fit(x, y)
    py = model.predict(tx)
    print("RandomForest RMSE = ", RMSE(ty, py))

def randomForestRegression(x, y, tx, ty):
    model = RandomForestRegressor(n_estimators = 190, max_depth = 15, max_leaf_nodes = 500)
    model.fit(x, y)
    py = model.predict(tx)
    print("RandomForestRegression RMSE = ", RMSE(ty, py))


def hyperparameterTuning(x, y, tx, ty):
    # Use GridSearchCV to tune all kinds of hyperparameters
    n = range(80, 220, 10) #n = range(10, 200, 10)
    hp = {'n_estimators':n}
    model = GridSearchCV(RandomForestRegressor(), hp, n_jobs=-1)
    model.fit(x, y)
    py = model.predict(tx)
    print("Best RandomForestRegression RMSE = ", RMSE(ty, py))
    print("Best hyper parameter", model.best_params_)


def MLPRegression(x, y, tx, ty):
    model = MLPRegressor(hidden_layer_sizes=(100, 5, 100), activation = "tanh", solver = "adam")
    model.fit(x, y)
    py = model.predict(tx)
    print("MLPRegression RMSE = ", RMSE(ty, py))

def adaBoostRegression(x, y, tx, ty):
    model = AdaBoostRegressor()
    model.fit(x, y)
    py = model.predict(tx)
    print("AdaBoostRegression RMSE = ", RMSE(ty, py))

# X_train, X_test, y_train, y_test = train_test_split(training_X, training_Y, test_size=0.1, random_state=20)
# print(X_train.shape[:])
# print(X_test.shape[:])
# print(y_train.shape[:])
# print(y_test.shape[:])
# randomForestClassify(X_train, y_train, X_test, y_test)

# decisionTree(X_train, y_train, X_test, y_test)
# lightGBMClassify(X_train, y_train, X_test, y_test)
# XGBoostClassify(X_train, y_train, X_test, y_test)
# randomForestClassify(X_train, y_train, X_test, y_test)
# randomForestRegression(X_train, y_train, X_test, y_test)
# MLPRegression(X_train, y_train, X_test, y_test)
# adaBoostRegression(X_train, y_train, X_test, y_test)

# decisionTree(training_X, training_Y, testing_X, testing_Y)
# lightGBMClassify(training_X, training_Y, testing_X, testing_Y)
# XGBoostClassify(training_X, training_Y, testing_X, testing_Y)
# randomForestClassify(training_X, training_Y, testing_X, testing_Y)
# randomForestRegression(training_X, training_Y, testing_X, testing_Y)
# MLPRegression(training_X, training_Y, testing_X, testing_Y)
# adaBoostRegression(training_X, training_Y, testing_X, testing_Y)

# hyperparameterTuning(training_X, training_Y, testing_X, testing_Y)