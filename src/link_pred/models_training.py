from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_curve
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchmetrics
from sklearn.neural_network import MLPClassifier


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



def get_best_MLP(train_samples_scaled, train_labels, validation_samples_scaled, validation_labels, gpu = False, verbose = 0):

    activation_list = ["logistic", "tanh", "relu"]
    learning_rate_list = ["constant","adaptive"]
    solver_list = ["lbfgs",'sgd','adam']
    best = 0.0
    best_model = None
    print(f"{'activation':^7} | {'learning_rate':^7} | {'solver':^7} | {'Thresh':^12} | {'F1':^9} ")
    for activation in activation_list:
        for learning_rate in learning_rate_list:
            for solver in solver_list:
                clf = MLPClassifier(hidden_layer_sizes=(train_samples_scaled.shape[1],int(train_samples_scaled.shape[1]/2)), learning_rate = learning_rate , activation=activation,solver=solver, random_state = 42)
                clf.fit(train_samples_scaled, train_labels)
                y_pred = clf.predict_proba(validation_samples_scaled)
                probas = np.array(y_pred)[:,1]
                fpr, tpr, thresh_list = roc_curve(validation_labels, probas)

                # obtain best thresh
                best_thresh = 0.5
                best_f1 = -np.inf
                for thresh in thresh_list:
                    f1 = f1_score(validation_labels, np.int32(probas>thresh))
                    if f1 >= best_f1:
                        best_f1 = f1
                        best_thresh = thresh

                if best_f1 >= best:
                    best_params = (activation, learning_rate, solver, best_thresh)
                    best_model = clf 

                print(f"{best_params[0]:^7} | {best_params[1]:^7} | {best_params[2]:^7} | {best_params[3]:^12} | {best_f1:^9} ")

    print("best params:")
    print(f"{best_params[0]:^7} | {best_params[1]:^7} | {best_params[2]:^7} | {best_params[3]:^12} | {best:^9} ")
    return(best_model, best_params)



def get_xgb(train_samples_scaled, train_labels, validation_samples_scaled, validation_labels, gpu = False, verbose = 0):
    n_estim = 2000
    if gpu:
        tree_method = 'gpu_hist'
        predictor="gpu_predictor"
    else:
        tree_method = 'hist'
        predictor="cpu_predictor"

    clf = XGBClassifier(n_estimators=n_estim, n_jobs=-1, tree_method=tree_method, predictor=predictor, random_state=42, seed=42)
    clf.fit(train_samples_scaled, train_labels, eval_metric="auc", early_stopping_rounds=10, eval_set=[(validation_samples_scaled, validation_labels)], verbose=verbose)
    y_pred = clf.predict_proba(validation_samples_scaled)
    probas = np.array(y_pred)[:,1]
    fpr, tpr, thresh_list = roc_curve(validation_labels, probas)

    # obtain best thresh
    best_thresh = 0.5
    best_acc = -np.inf
    for thresh in thresh_list:
        acc = accuracy_score(validation_labels, np.int32(probas>=thresh))
        if acc >= best_acc:
            best_acc = acc
            best_thresh = thresh

    acc = accuracy_score(validation_labels, probas>=best_thresh)
    f1 = f1_score(validation_labels, probas>=best_thresh)       
    
    print(f'accuracy: {acc}')
    print(f'f1: {f1}')
    print(f'best_thresh: {thresh}')

    
    return(clf, best_thresh)




###### neural networks corner
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchmetrics

class Net(nn.Module):
    def __init__(self, input_size, output_size = 2):
        super(Net, self).__init__()
        self.net = nn.Sequential(
                    nn.Linear(input_size, int(input_size/2)),
                    nn.ReLU(),
                    nn.BatchNorm1d(int(input_size/2)),
                    nn.Linear(int(input_size/2), int(input_size/4)),
                    nn.ReLU(),
                    nn.BatchNorm1d(int(input_size/4)),
                    nn.Linear(int(input_size/4), output_size)
                    )
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, x):
        x = self.net(x)
        return self.softmax(x)

class Dataset(Dataset):
  def __init__(self,set_df, labels):
    self.x=torch.tensor(set_df,dtype=torch.float32)
    self.y=torch.tensor(labels,dtype=torch.long)
  def __len__(self):
    return len(self.y)
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]


def train_one_epoch(model, trainloader, validationloader, optimizer, device): 
    losses = []
    val_auroc = torchmetrics.AUROC(num_classes = 2)
    val_f1 = torchmetrics.F1Score(num_classes = 2)
    train_f1 = torchmetrics.F1Score(num_classes = 2)
    model.train()
    for (features, target) in tqdm(trainloader):
        features, target = features.to(device), target.to(device)
        optimizer.zero_grad()
        predictions = model(features)
        predicted_classes = torch.argmax(predictions, dim=1)
      
        criterion = nn.CrossEntropyLoss()
        loss = criterion(predictions, target)
        losses.append(float(loss))
        loss.backward()
        optimizer.step()
        f1_train = train_f1(predicted_classes.cpu(), target.cpu())

    model.eval()
    with torch.no_grad():
      for (features, target) in (validationloader):
          features, target = features.to(device), target.to(device)

          predictions = model(features)
          predicted_classes = torch.argmax(predictions, dim=1)
          
          validation_auroc = val_auroc(predictions.cpu(), target.cpu())
          f1_val = val_f1(predicted_classes.cpu(), target.cpu())

    print("NN")
          
    print('average train loss: ', np.mean(losses))

    print('validation f1: ', val_f1.compute())
    print('train f1: ', train_f1.compute())
    print('validation auroc: ', val_auroc.compute())

    return


def get_neural_net(train_samples_scaled, train_labels, validation_samples_scaled, validation_labels, device = 'cuda'):
    training_set = Dataset(train_samples_scaled, train_labels)
    validation_set = Dataset(validation_samples_scaled, validation_labels)
    train_loader = DataLoader(training_set, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=64, shuffle=True)
    clf = Net(input_size = train_samples_scaled.shape[1]).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr = 0.1)
    for epoch in range(0, 5):
        train_one_epoch(clf , train_loader, validation_loader, optimizer,device)
    return(clf)



#### MLP with sklearn
