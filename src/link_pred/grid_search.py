from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score 
from xgboost import XGBClassifier

def get_best_xgb(train_samples_scaled, train_labels, validation_samples_scaled, validation_labels):

    n_estim = 2000
    max_depth_candidates = [10, 15, 20]
    learning_rates = [0.1, 0.05, 0.01]
    min_child_weights = [1, 3, 4, 5]
    best = 0.0
    best_model = None
    print(f"{'N_Estimators':^7} | {'Max_Depth':^7} | {'Min Child Weights':^7} | {'Learning_Rate':^7} | {'Accuracy':^12} | {'F1':^9} ")

    for max_depth in tqdm(max_depth_candidates):
        for lr in learning_rates:
            for min_child_weight in min_child_weights:
                clf = XGBClassifier(max_depth=max_depth, learning_rate=lr, min_child_weight=min_child_weight, n_estimators=n_estim, n_jobs=4, tree_method='gpu_hist', predictor="gpu_predictor", random_state=42, seed=42)
                clf.fit(train_samples_scaled, train_labels, eval_metric="auc", early_stopping_rounds=300, eval_set=[(validation_samples_scaled, validation_labels)], verbose=0)
                y_pred = clf.predict(validation_samples_scaled)
                acc = accuracy_score(validation_labels, y_pred)
                f1 = f1_score(validation_labels, y_pred)
                if f1 > best:
                    best_params = (clf.best_ntree_limit, max_depth, lr, acc, f1)
                    best = f1
                    best_model  = clf
                print(f"{clf.best_ntree_limit:^7} | {max_depth:^7} | {min_child_weight:^7} | {lr:^7} | {acc:^12} | {f1:^9}")
            ############
            print()
            print(f"Best Params:")
            print(f"{best_params[0]:^7} | {best_params[1]:^7} | {best_params[2]:^7} | {best_params[3]:^12} | {best_params[4]:^9}")
    return(best_model)