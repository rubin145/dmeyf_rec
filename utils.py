import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import lightgbm as lgb
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import make_scorer

def calculo_ganancia_registro(test_record,pred_record):
    if pred_record:
        if test_record:
            return 78000
        else:
            return -2000
    else:
        return 0

calculo_ganancia_arr = np.frompyfunc(calculo_ganancia_registro,2,1) #recibe dos arrays (test,pred) y devuelve un array de ganancias individuales

def ganancia_total_np(y_test, y_pred):
    return sum(calculo_ganancia_arr(y_test,y_pred))

#esta está porque la versión con np.frompyfunc falla (couldn't pickle task con cross_val_score)
def ganancia_total(y_test,y_pred):
    ganancias = []
    for t,p in zip(y_test, y_pred):
        if p:
            if t:
                ganancias.append(78000)
            else:
                ganancias.append(-2000)
    return sum(ganancias)

#con esta dif sirve si devuelve los scores.
def ganancia_total_score(y_test,y_pred):
    ganancias = []
    for t,p in zip(y_test, y_pred):
        if p > 0.025:
            if t:
                ganancias.append(78000)
            else:
                ganancias.append(-2000)
    return sum(ganancias)

def ganancia_total_eval(y_test,y_pred): #ver Custom eval function note https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.fit
    return 'ganancia', ganancia_total(y_test,y_pred), True

#pareciera que en la api nativa de lgbm están al revés los argumentos. feval en https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.cv.html
def ganancia_total_eval_nativa(preds,eval_data):
    y_pred = preds.round()
    return 'ganancia', ganancia_total(eval_data.label,y_pred), True

g_score_base = make_scorer(ganancia_total,greater_is_better=True)

g_score = make_scorer(ganancia_total_score, greater_is_better=True, needs_proba=True)


def semillerio_bools(size, n_seeds, X, y, new_X, feature_names, estimator, params):
    preds = np.empty((size,n_seeds))
    for i in range(1,n_seeds+1):
        params['random_state'] = i
        model = estimator(**params)
        model.fit(X, y, feature_name = feature_names)
        scores = model.predict_proba(new_X)
        y_pred = scores[:,1] > 0.025
        preds[:,i-1] = y_pred
        
    return np.mean(preds,axis=1).round().astype(bool) #promedia los True/False


def semillerio_scores(size, n_seeds, X, y, new_X, feature_names, estimator, params):
    all_scores = np.empty((size,n_seeds))
    for i in range(1,n_seeds+1):
        params['random_state'] = i
        model = estimator(**params)
        model.fit(X, y, feature_name = feature_names)
        scores = model.predict_proba(new_X)
        all_scores[:,i-1] = scores[:,1]
    preds = np.mean(all_scores,axis=1) > 0.025 #acá promedio los scores y después paso a clase
    return preds.astype(bool)



# #función a maximizar en la OB
# def black_box_function(params_var, params_fij, X_train, y_train, X_test, y_test):
#     model = lgb.LGBMClassifier(**params)
#     model.fit(X_train, y_train)
#     scores = model.predict_proba(X_test)
#     y_pred = scores[:,1] > 0.025
#     return ganancia_total(y_test,y_pred)

# def black_box_function(leaf_size_log, coverage, learning_rate, feature_fraction):
#     full_params = {**params_fijos,'learning_rate':learning_rate, 'feature_fraction':feature_fraction}
#     full_params['min_data_in_leaf'] = np.maximum(1, np.around(X_train.shape[0] / (2.0 ** leaf_size_log)).astype(int))
#     full_params['num_leaves']  =  np.minimum(131072, np.maximum(2, np.around(coverage * X_train.shape[0] / full_params['min_data_in_leaf']).astype(int)))
#     #que carajo es 131072
#     #del params['leaf_size_log']
#     #del params['coverage']
#     #full_params = {**params,**params_fijos}
#     model = lgb.LGBMClassifier(**full_params)
#     return cross_val_score(model, X_train, y_train, scoring=g_score, cv=cv, n_jobs=-1).mean()
#     #acá se pierde la varianza de los scores pero bueno...


