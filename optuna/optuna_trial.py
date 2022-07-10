import pandas as pd
import numpy as np
from optuna.multi_objective import trial
from sklearn import ensemble, metrics, model_selection
from sklearn import decomposition , preprocessing , pipeline
from skopt import gp_minimize
from functools import partial
from skopt import space
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
import optuna
import os


'''def optimizeh(params, x, y):
    print(params)
    model = ensemble.RandomForestClassifier(**params)
    kf = model_selection.StratifiedKFold(n_splits=7)
    accuracies = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_acc)
    return -1.0 * np.mean(accuracies)'''

def optimizeh(trial,x,y):
    criterion = trial.suggest_categorical("criterion",["gini","entropy"])
    n_estimators = trial.suggest_int("n_estimators",100,1500)
    max_depth = trial.suggest_int("max_depth",3,15)
    max_features = trial.suggest_uniform("max_features",0.01,1.0)


    model = ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        criterion=criterion
    )
    kf = model_selection.StratifiedKFold(n_splits=7)
    accuracies = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_acc)
    return -1.0 * np.mean(accuracies)

if __name__ == "__main__":
    df = pd.read_csv('./input/mobile_prices1.csv')
    x = df.drop(["price_range"], axis=1).values
    y = df.price_range.values

    optimization_function = partial(optimizeh, x=x, y=y)

    study = optuna.create_study(direction="minimize")
    study.optimize(optimization_function,n_trials=16)

    '''param_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 3, 15, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 600, 1)),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "max_features": hp.uniform("max_features", 0.01, 1)
    }

    optimization_function = partial(
        optimizeh,
        x=x,
        y=y
    )
    trials = Trials()

    result = fmin(
        fn=optimization_function,
        space=param_space,
        algo=tpe.suggest,
        max_evals=17,
        trials=trials,
    )
    print(result)'''




    '''df = pd.read_csv('./input/mobile_prices1.csv')

    def optimize(params,param_names,x,y):
        params = dict(zip(param_names,params))
        model = ensemble.RandomForestClassifier(**params)
        kf = model_selection.StratifiedKFold(n_splits=7)
        accuracies = []
        for idx in kf.split(X=x,y=y):
            train_idx, test_idx = idx[0] , idx[1]
            xtrain = x[train_idx]
            ytrain = y[train_idx]

            xtest = x[test_idx]
            ytest = y[test_idx]

            model.fit(xtrain, ytrain)
            preds = model.predict(xtest)
            fold_acc = metrics.accuracy_score(ytest, preds)
            accuracies.append(fold_acc)

        return -1.0 * np.mean(accuracies)

    x = df.drop(["price_range"], axis=1).values
    y = df.price_range.values

    param_space = [
        space.Integer(3, 15, name="max_depth"),
        space.Integer(100, 600, name="n_estimators"),
        space.Categorical(["gini", "entropy"], name="criterion"),
        space.Real(0.01, 1, name="max_features", prior="uniform")
    ]

    param_names = [
        "max_depth",
        "n_estimators",
        "criterion",
        "max_features"
    ]

    optimization_function = partial(
        optimize,
        param_names = param_names,
        x = x,
        y = y
    )

    result = gp_minimize(
        optimization_function,
        dimensions = param_space,
        n_calls = 17,
        n_random_starts = 12,
        verbose = 10
    )

    print(dict(zip(param_names, result.x)))'''







