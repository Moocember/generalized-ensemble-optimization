import pandas as pd
import numpy as np
from hyperopt import hp,fmin, tpe,STATUS_OK,space_eval
from Load_Data import load_numerai_data,pickleIt
from functools import partial
from ML import SGDClassify,XGB,SVC,LogisticRegression,Sigmoid,UniformVote
import pickle



def set_seed(params):
    """
    Set random seed if it exists in params.

    Args:
        params: Model hyperparameters. Extracted from childern of the ML_Model class.

    Returns:
        Params stripped of the seed key.
    """
    seed_params = params["param"].copy()
    if 'seed' in params["param"].keys():
        np.random.seed(int(seed_params.pop('seed')))
    return seed_params

def subset_keys(dictionary,letter):
    """
    Subset dictionary keys beginning with a letter.

    Args:
        dictionary: Dictionary with keys to subset.
        letter: First letter to subset.

    Returns:
        Subset of keys.
    """
    all_keys = list(dictionary.keys())

    return [idx for idx in all_keys if idx[0] == letter]

def validation_classification(model,data):
    """
        Use trained model to classify.

        Args:
            model: Already trained model.
            data: Train, validation and test data.

        Returns:
            Dictionary of predicted values for train, validation and test sets.
    """
    x_keys = subset_keys(data,'x')

    output = [model.predict_proba(data[x])[:, 1] for x in x_keys]

    return dict(zip(x_keys, output))

def classification(params,data):
    """
    Generalized function to fit and classify data.

    Args:
        params: Model hyperparameters. Extracted from childern of the ML_Model class.
        data: Train, validation and test data.

    Returns:
        Dictionary of predicted values for train, validation and test sets.
    """

    seed_params = set_seed(params)

    model = params["algo"](**seed_params)

    model.fit(data['x_train'], data['y_train'])

    return validation_classification(model, data)

def binaryAccuracy(probabilities,y_validation):
    """
    Tests classification accuracy.

    Args:
        probabilities:Probabilities each row are in either class.
        y_validation: Vector of true classes.

    Returns:
        Classification accuracy (AUC).
    """
    correct = np.round(probabilities) == y_validation

    return np.mean(correct)

def fitness(params, data):
    """
    Calculate binary accuracy of a given model and parameter set.

    Args:
        params: Model hyperparameters. Extracted from childern of the ML_Model class.
        data: Train, validation and test data.

    Returns:
        Negative classification accuracy (AUC).
        Negative because hyperopt minimizes fitness function.
    """
    probabilities = classification(params,data)['x_validation']

    AUC = binaryAccuracy(probabilities, data['y_validation'])

    print("- accuracy: ", AUC)
    return -AUC

def objective(params, data):
    """
    Wrapper for compatibility with Hyperopt's fmin function.

    Args:
        params: Model hyperparameters. Extracted from childern of the ML_Model class.
        data: Train, validation and test data.

    Returns:
        Dictionary of loss and status.
    """
    return {'loss': fitness(params,data), 'status': STATUS_OK}

def find_best_parameters(objective,model,data,algo=tpe.suggest,max_evals=100):
    """
    Wrapper to call Hyperopt's fmin function.
    Fmin uses expected improvement algo to find optimal hyperparameters any model.


    Args:
        objective: objective function defined on line 76.
        model: Model extracted from classes defined in ML.py.
        data: Train, validation and test data.
        algo: fmin hyperparameter, see: https://github.com/hyperopt/hyperopt/wiki/FMin
        max_evals: Number of models to test before returning.

    Returns:
        Optimal set of hyperparemeters for the model.
    """
    params = model.__dict__
    if len(model.optimized) > 0:
        objFunc = partial(objective, data=data)

        bestParams = fmin(objFunc,  params, algo=algo, max_evals=max_evals)

        return space_eval(params, bestParams)
    else:
        return params

def initalize_empty_dfs(keys,columns):
    empty_dfs = [pd.DataFrame(columns=columns) for x in keys]
    return dict(zip(keys, empty_dfs))

def find_best_models(objective,models,data,algo=tpe.suggest,max_evals=100):
    """
    Find optimal set of hyperparameters for multiple models.

    Args:
        objective: objective function defined on line 76.
        model: Model extracted from classes defined in ML.py.
        data: Train, validation and test data.
        algo: fmin hyperparameter, see: https://github.com/hyperopt/hyperopt/wiki/FMin
        max_evals: Number of models to test before returning.

    Returns:
        Optimal set of hyperparemeters for the model.
    """
    model_set = dict()
    for key in models.keys():
        model_set[key] = find_best_parameters(objective, models[key], data, algo=algo, max_evals=max_evals)
    return model_set


def classify_models(models,data):
    """
    Perform classification on multiple models and return a dictonary of Dataframes.

    Args:
        objective: objective function defined on line 76.
        model: Model extracted from classes defined in ML.py.
        data: Train, validation and test data.
        algo: fmin hyperparameter, see: https://github.com/hyperopt/hyperopt/wiki/FMin
        max_evals: Number of models to test before returning.

    Returns:
        Optimal set of hyperparemeters for the model.
    """
    x_keys = subset_keys(data,'x')
    df_dict = initalize_empty_dfs(x_keys,models.keys())
    counter = 0
    for key in models.keys():
        counter+=1
        print(counter)
        probabilities = classification(models[key], data)
        for x in x_keys:
            df_dict[x][key] = probabilities[x]

    return df_dict



def blend_models(objective,models,metamodel,data,max_evals=100):
    """
    Finds optimal hyperparameters for each model individually.

    Args:
        objective: objective function defined on line 76.
        model: Model extracted from classes defined in ML.py.
        data: Train, validation and test data.
        algo: fmin hyperparameter, see: https://github.com/hyperopt/hyperopt/wiki/FMin
        max_evals: Number of models to test before returning.

    Returns:
        Optimal set of hyperparemeters for the model.
    """

    best_models = find_best_models(objective,models,data,algo=tpe.suggest,max_evals=max_evals)

    probabilities = classify_models(best_models,data)

    data.update(probabilities)

    probabilities['params'] = find_best_parameters(objective, metamodel,data, algo=tpe.suggest, max_evals=max_evals)
    probabilities['AUC_validation'] = -fitness(probabilities['params'],data)
    return probabilities



import matplotlib.pyplot as plt
if __name__ == '__main__':

    data = load_numerai_data()

    #knn = KNN(n_neighbors = hp.choice('nn', np.arange(1, 101,2, dtype=int)))
    sgd = SGDClassify(alpha = hp.uniform('alpha',0,1),
                      max_iter = hp.uniform('max_iter',100,5000))

    xgb = XGB(n_estimators= hp.choice('n_estimators', np.arange(100, 1000, dtype=int)),
              eta = hp.quniform('eta', 0.025, 0.5, 0.025),
              max_depth = hp.choice('max_depth', np.arange(1, 14, dtype=int)),
              min_child_weight = hp.quniform('min_child_weight', 1, 6, 1),
              subsample = hp.quniform('subsample', 0.5, 1, 0.05),
              gamma = hp.quniform('gamma', 0.5, 1, 0.05),
              colsample_bytree = hp.quniform('colsample_bytree', 0.5, 1, 0.05))

    svc = SVC(C = hp.lognormal('C',0,20),
              max_iter=hp.uniform('max_iter', 100, 5000))

    models = {'SGD': sgd,
              'SVC': svc,
              'XGB': xgb}
    """
    blender = LogisticRegression(C = hp.uniform('C',0,1),
                            max_iter = hp.uniform('max_iter',100,1000))
    """

    blender = UniformVote()
    blendedProbabilities = blend_models(objective,models,blender,data,max_evals=50)

    print(blendedProbabilities['AUC_validation'])
    pickleIt(blendedProbabilities, 'blend_params2')