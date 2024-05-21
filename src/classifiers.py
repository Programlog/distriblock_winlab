import torch.nn as nn
from .tools import merge, largest_within_delta, smaller_within_delta_indx, perf_measure, dict_list
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc
import numpy as np
import torch
import tqdm
import os
import torch.optim as optim
from sklearn.model_selection import train_test_split
import copy


class Deep(nn.Module):
    """
    Neural Network arquitecture.
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(24, 72)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(72, 72)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(72, 72)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(72, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

def distriblock_gaussians(train_set, test_set, adv_set, key):
    """
    Fit a Gaussian distribution to each Characteristic score computed for the utterances from a training set of benign data. 
    If the probability of a new audio sample is below a chosen threshold under the Gaussian model, 
    this example is classified as adversarial.

    :param train_set: Training set of benign data.
    :param test_set: Testing set of benign data.
    :param adv_set: Testing set of adversarial data.
    :param key: Characteristic to fit the gaussian.
    :return: Classifier performance in terms of AUROC.
    """
    char_key = []
    char_key.append(key)
    char_key.append("benign_flg")
    test_metrics = {x: test_set[x] for x in char_key}      
    adv_metrics = {x: adv_set[x] for x in char_key}
    test_all = merge(test_metrics, adv_metrics, char_key)
    mean, std = norm.fit(train_set[key])

    fitted_norm = norm.pdf(test_all[key], loc=mean, scale=std + 1e-08)
    fpr, tpr, threshold = roc_curve(test_all['benign_flg'], fitted_norm)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def distriblock_ensembles(train_set, test_set, val_set, adv_set, adv_val_set, ensemble_chars):
    """
    Employ ensemble models (EMs), in which multiple Gaussian distributions, fitted to a single score each, 
    produce a unified decision by a majority vote. 

    :param train_set: Training set of benign data.
    :param val_set: Validating set of benign data.
    :param test_set: Testing set of benign data.
    :param adv_val_set: Validating set of adversarial data.
    :param adv_set: Testing set of adversarial data.
    :param ensemble_chars: List of Characteristics.
    :return: Classifier performance in terms of Acc, TP, FP, TN, FN, FPR, TPR, precision, recall, and F1 score.
    """
    cont = 0
    majority_vote = np.array([0] * 2 * len(train_set["benign_flg"]))
    for i in ensemble_chars:
        val_metrics = {x: val_set[x] for x in [i, "benign_flg"]}           
        adv_val_metrics = {x: adv_val_set[x] for x in [i, "benign_flg"]}
        val_guassian_metrics = merge(val_metrics, adv_val_metrics, [i, "benign_flg"])
        mean, std = norm.fit(train_set[i])
        thres_fitted_norm = norm.pdf(val_guassian_metrics[i], loc=mean, scale=std + 1e-08)
        fpr, tpr, thresholds = roc_curve(val_guassian_metrics['benign_flg'], thres_fitted_norm, drop_intermediate=False)
        indx_tpr = largest_within_delta(tpr, 0.5)
        indx = smaller_within_delta_indx(fpr, indx_tpr, 0.01)
        thres = thresholds[indx]
        # '''
        test_metrics = {x: test_set[x] for x in [i, "benign_flg"]}  
        adv_metrics = {x: adv_set[x] for x in [i, "benign_flg"]}
        test_guassian_metrics = merge(test_metrics, adv_metrics, [i, "benign_flg"])
        fitted_norm = norm.pdf(test_guassian_metrics[i], loc=mean, scale=std + 1e-08)
        majority_vote += (fitted_norm >= thres)
        cont+=1
        # '''
    y_actual = np.array(test_guassian_metrics['benign_flg'])
    y_hat = (majority_vote >= cont/2)
    TP, FP, TN, FN = perf_measure(y_actual, y_hat)

    TPR = TP / (TP + FN)
    FPR = FP / (TN + FP)
    acc = (y_hat == y_actual) 
    precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = TP / (TP + (FN + FP)/2)
    return np.mean(acc), TP, FP, TN, FN, FPR, TPR, precision, Recall, F1

def distriblock_NN(train_set, test_set, adv_set, adv_val_set, model_path):
    """
    Construct an NN that takes all the Characteristics as input.

    :param train_set: Training set of benign data.
    :param test_set: Testing set of benign data.
    :param adv_val_set: Validating set of adversarial data.
    :param adv_set: Testing set of adversarial data.
    :param model_path: *.pth NN model file path.
    :return: Classifier performance in terms of AUROC, Acc, TP, FP, TN, FN, FPR, TPR, precision, recall, and F1 score.
    """
    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 
    device = torch.device(dev)
    model = Deep().to(device)
    if not os.path.exists(model_path):     
        clean_train_data = dict_list(train_set)
        adv_train_data = dict_list(adv_val_set)
        train_dataset = torch.tensor(clean_train_data + adv_train_data, dtype=torch.float32).to(device)
        X = train_dataset[:, 0:24]
        y = train_dataset[:, 24].reshape(-1, 1)
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, shuffle=True)
        acc = model_train(model, X_train, y_train, X_val, y_val)
        torch.save(model.state_dict(), model_path)
        print(f"Validation Accuracy: {acc*100:.2f}%")

    model.load_state_dict(torch.load(model_path))
    model.eval()
    clean_test_data = dict_list(test_set)
    adv_test_data = dict_list(adv_set)
    test_dataset = torch.tensor(clean_test_data + adv_test_data, dtype=torch.float32).to(device)
    r=torch.randperm(200)
    test_dataset = test_dataset[r]
    X_test = test_dataset[:, 0:24]
    y_test = test_dataset[:, 24].reshape(-1, 1)
    with torch.no_grad():
        y_pred = model(X_test)
        fpr, tpr, thresholds = roc_curve(y_test.cpu().numpy(), y_pred.cpu().numpy(), drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        
        acc = (y_pred.round() == y_test).float().mean()
        y_actual = y_test
        y_hat = y_pred.round()
        TP, FP, TN, FN = perf_measure(y_actual, y_hat)
        TPR = TP / (TP + FN)
        FPR = FP / (TN + FP)
        precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = TP / (TP + (FN + FP)/2)
    return roc_auc, acc, TP, FP, TN, FN, FPR, TPR, precision, Recall, F1 

def model_train(model, X_train, y_train, X_val, y_val):
    """
    Training a NN.

    :param X_train: Data for training set.
    :param y_train: Labels for training set.
    :param X_val: Data for validation set.
    :param y_val: Labels for validation set.
    :return: Best performance accuracy based on the validation set. 
    """
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
 
    n_epochs = 250   # number of epochs to run
    batch_size = 3 # 10  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)
 
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
    print("------------------------- Training Neural Network model...")
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc

def gaussian_filtering(train_set, test_set, val_set, adv_set, adv_val_set):
    """
    Fit a Gaussian distribution to each filtering method using the WER metric from a training set of benign data. 
    If the probability of a new audio sample is below a chosen threshold under the Gaussian model, 
    this example is classified as adversarial.

    :param train_set: Training set of benign data.
    :param val_set: Validating set of benign data.
    :param test_set: Testing set of benign data.
    :param adv_val_set: Validating set of adversarial data.
    :param adv_set: Testing set of adversarial data.
    :return: Classifier performance in terms of Acc, TP, FP, TN, FN, FPR, TPR, precision, recall, and F1 score.
    """
    val_data = val_set[0]   
    adv_val_data = adv_val_set[0]
    val_flg = val_set[1]   
    adv_val_flg = adv_val_set[1]
    val_guassian_data = val_data + adv_val_data
    val_guassian_flg = val_flg + adv_val_flg
    mean, std = norm.fit(train_set[0])
    thres_fitted_norm = norm.pdf(val_guassian_data, loc=mean, scale=std + 1e-08)
    fpr, tpr, thresholds = roc_curve(val_guassian_flg, thres_fitted_norm, drop_intermediate=False)

    indx_tpr = largest_within_delta(tpr, 0.5)
    indx = smaller_within_delta_indx(fpr, indx_tpr, 0.01)
    thres = thresholds[indx]

    test_data = test_set[0] 
    adv_data = adv_set[0]
    test_flg = test_set[1] 
    adv_flg = adv_set[1]
    test_guassian_data = test_data + adv_data
    test_guassian_flg = test_flg + adv_flg
    fitted_norm = norm.pdf(test_guassian_data, loc=mean, scale=std + 1e-08)
   
    y_actual = test_guassian_flg
    y_hat = (fitted_norm >= thres)
    TP, FP, TN, FN = perf_measure(y_actual, y_hat)

    TPR = TP / (TP + FN)
    FPR = FP / (TN + FP)
    acc = (y_hat == y_actual) 
    precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = TP / (TP + (FN + FP)/2)
    return np.mean(acc), TP, FP, TN, FN, FPR, TPR, precision, Recall, F1

