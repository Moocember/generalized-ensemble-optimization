from sklearn import linear_model,svm,neighbors
from xgboost import XGBClassifier
from hyperopt import hp
import numpy as np
class Ensemble:
    """
    Class which contains all models to be ensembled.

    Args:
        models (dict): Dictionary containing all model classes to be ensembled.

    Attributes:
        __dict__ (dict): Dictionary of dictionaries of hyperparameters for each model.
    """
    def __init__(self,models):
        dicts = []
        for key in models.keys():
            dicts.append(models[key].__dict__)
        self.models = hp.choice('models',dicts)

class ML_Model:
    seed = hp.uniform('seed', 0,999999999)
    def __init__(self):
        pass

    def which_optimized_hyperparameters(self):
        return [key for key, value in self.__dict__.items() if type(value) == type(self.seed)]

    def general_parameters(self,seed = seed):
        self.seed = seed
        self.param = self.__dict__.copy()
        self.optimized = self.which_optimized_hyperparameters()

class SGDClassify(ML_Model):
    """
    Class which contains stocastic gradient descent classifier.

    Args:
        To read about each hyperparemeter read:
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

    Attributes:
        __dict__ (dict): Dictionary of model hyperparameters.
    """
    tol = 0
    loss = 'log'
    penalty = 'l2'
    n_jobs =  -1

    def __init__(self, alpha, max_iter,tol = tol, loss = loss, penalty = penalty, n_jobs = n_jobs):
      self.alpha = alpha
      self.max_iter = max_iter
      self.tol = tol
      self.loss = loss
      self.penalty = penalty
      self.n_jobs = n_jobs

      super(SGDClassify, self).general_parameters()
      self.algo = linear_model.SGDClassifier

class XGB(ML_Model):
    """
    Class which contains Extreme Gradient Boosting classifier.

    Args:
        To read about each hyperparemeter read:
        https://xgboost.readthedocs.io/en/latest/parameter.html

    Attributes:
        __dict__ (dict): Dictionary of model hyperparameters.
    """
    eval_metric = 'auc'
    objective = 'binary:logistic'
    nthread = -1
    booster = 'gbtree'
    tree_method = 'exact'

    def __init__(self,n_estimators,eta,max_depth,min_child_weight,subsample,gamma,colsample_bytree,eval_metric=eval_metric,objective=objective,nthread=nthread,booster=booster,tree_method=tree_method):

        self.n_estimators= n_estimators
        self.eta = eta
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.gamma = gamma
        self.colsample_bytree = colsample_bytree
        self.eval_metric = eval_metric
        self.objective = objective
        self.nthread = nthread
        self.booster = booster
        self.tree_method = tree_method

        super(XGB, self).general_parameters()
        self.algo = XGBClassifier


class SVC(ML_Model):
    """
    Class which contains support vector classification.

    Args:
        To read about each hyperparemeter read:
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    Attributes:
        __dict__ (dict): Dictionary of model hyperparameters.
    """
    probability = True
    kernel = 'linear'
    def __init__(self,C,max_iter,kernel=kernel,probability=probability):
        self.probability = probability
        self.C = C
        self.max_iter = max_iter
        self.kernel = kernel

        super(SVC, self).general_parameters()
        self.algo = svm.SVC

class Sigmoid(ML_Model):
    scalar = 5
    posOnly = True
    def __init__(self,scalar = scalar, posOnly = posOnly):
        self.scalar = scalar
        self.posOnly = posOnly

        super(Sigmoid, self).general_parameters()
        self.algo = Sigmoid

    def fit(self,x,y):
        pass

    def predict_proba(self,x):
        if self.posOnly:
            fx = 1 / (1 + np.exp(-self.scalar * x + self.scalar/2))
        else:
            fx = 1 / (1 + np.exp(-self.scalar * x))
        avgSigmoid = np.mean(fx,axis=1)
        return np.c_[np.ones((len(avgSigmoid),1)),avgSigmoid]

class UniformVote(ML_Model):
    scalar = 5
    posOnly = True
    def __init__(self,scalar = scalar, posOnly = posOnly):
        self.scalar = scalar
        self.posOnly = posOnly

        super(UniformVote, self).general_parameters()
        self.algo = UniformVote

    def fit(self,x,y):
        pass

    def predict_proba(self,x):
        binaryVals = np.mean(np.round(x), axis=1)
        fx =  np.round(binaryVals)
        return np.c_[np.ones((len(fx),1)),fx]

def uniformVote(x):
    binaryVals = np.mean(np.round(x),axis=1)
    return np.round(binaryVals)

class LogisticRegression(ML_Model):
    """
    Class which contains Logistic Regression.

    Args:
        To read about each hyperparemeter read:
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    Attributes:
        __dict__ (dict): Dictionary of model hyperparameters.
        """
    solver = 'sag'
    tol = 0
    penalty = 'l2'
    n_jobs =  -1

    def __init__(self, C, max_iter, tol=tol, solver=solver, penalty=penalty, n_jobs=n_jobs):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.penalty = penalty
        self.n_jobs = n_jobs

        super(LogisticRegression, self).general_parameters()
        self.algo = linear_model.LogisticRegression
