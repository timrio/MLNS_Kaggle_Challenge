from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_curve
from xgboost import XGBClassifier
import numpy as np

def get_best_xgb(train_samples_scaled, train_labels, validation_samples_scaled, validation_labels):

    n_estim = 2000
    max_depth_candidates = [10, 15, 20]
    learning_rates = [0.1, 0.05, 0.01]
    min_child_weights = [1, 3, 4, 5]
    best = 0.0
    best_model = None
    best_thresh = None
    print(f"{'N_Estimators':^7} | {'Max_Depth':^7} | {'Min Child Weights':^7} | {'Learning_Rate':^7} | {'Thresh':^12} | {'Accuracy':^12} | {'F1':^9} ")

    for max_depth in max_depth_candidates:
        for lr in learning_rates:
            for min_child_weight in min_child_weights:
                clf = XGBClassifier(max_depth=max_depth, learning_rate=lr, min_child_weight=min_child_weight, n_estimators=n_estim, n_jobs=-1, tree_method='gpu_hist', predictor="gpu_predictor", random_state=42, seed=42)
                clf.fit(train_samples_scaled, train_labels, eval_metric="auc", early_stopping_rounds=10, eval_set=[(validation_samples_scaled, validation_labels)], verbose=0)
                y_pred = clf.predict_proba(validation_samples_scaled)
                probas = np.array(y_pred)[:,1]
                fpr, tpr, thresh_list = roc_curve(validation_labels, probas)

                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
                thresh = thresh_list[ix]

                acc = accuracy_score(validation_labels, probas>=thresh)
                f1 = f1_score(validation_labels, probas>=thresh)
                if f1 > best:
                    best_params = (clf.best_ntree_limit, max_depth, lr, acc, f1)
                    best = f1
                    best_model  = clf
                    best_thresh = thresh
                   
                    
                print(f"{best_params[0]:^7} | {best_params[1]:^7} | {best_params[2]:^7} |{best_thresh:^12} | {best_params[3]:^12} | {best_params[4]:^9}")
    ############
    print()
    print(f"Best Params:")
    print(f"{best_params[0]:^7} | {best_params[1]:^7} | {best_params[2]:^7} |{best_thresh:^12} | {best_params[3]:^12} | {best_params[4]:^9}")

    
    return(best_model, best_thresh)