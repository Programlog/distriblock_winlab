"""
DistriBlock script, a detection method for adversarial attacks on neural network-based ASR systems. 
Several characteristics of the distribution over the output tokens can serve as features of binary classifiers.
(https://arxiv.org/abs/2305.17000)
"""

import argparse
import os
from src.tools import load_meas_data
from src.classifiers import distriblock_gaussians, distriblock_ensembles, distriblock_NN
import sys


CHARACTERISTICS_EM = [["Median mean"], 
                    ["Median mean", "Entropy mean", "Entropy max"], 
                    ["Median mean", "Entropy mean", "Entropy max", "Entropy median", "Median max"],
                    ["Median mean", "Entropy mean", "Entropy max", "Entropy median", "Median max", "Max median", "Max mean"],
                    ["Median mean", "Entropy mean", "Entropy max", "Entropy median", "Median max", "Max median", "Max mean", "Min mean", "Max min"]
                    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("FOLDER_ORIG", help="Folder path that contains the characteristics of the benign examples.",
                        type=str, nargs=1)
    parser.add_argument("FOLDER_ADV", help="Folder path that contains the characteristics of the adversarial examples.",
                    type=str, nargs=1)

    orig_path = parser.parse_args().FOLDER_ORIG[0]
    adv_path = parser.parse_args().FOLDER_ADV[0]
    file_names = ["train.pickle", "test.pickle", "adv_test.pickle", "val.pickle", "adv_train.pickle"]

    if os.path.exists(f"{orig_path}/{file_names[0]}"):
        train_meas = load_meas_data(f"{orig_path}/{file_names[0]}", benign_flg=True)
        keys = []
        for i in train_meas:
            keys.append(i)
    if os.path.exists(f"{orig_path}/{file_names[1]}"):
        test_meas = load_meas_data(f"{orig_path}/{file_names[1]}", benign_flg=True)
    if os.path.exists(f"{orig_path}/{file_names[3]}"):
        val_meas = load_meas_data(f"{orig_path}/{file_names[3]}", benign_flg=True)    
    if os.path.exists(f"{adv_path}/{file_names[2]}"):
        adv_meas = load_meas_data(f"{adv_path}/{file_names[2]}", benign_flg=False)
    if os.path.exists(f"{adv_path}/{file_names[4]}"):
        adv_val_meas = load_meas_data(f"{adv_path}/{file_names[4]}", benign_flg=False)

    if (len(keys) - 1 == 24):
        print(" ")
        print("------------------------- Gaussian Classifiers results: ---------------------------")
        key_cont = 0
        for i in range(6):
            for k in range(4):
                auroc = distriblock_gaussians(train_meas, test_meas, adv_meas, keys[k + key_cont])
                print("Characteristic: \"{}\". AUROC: {:.4f}".format(keys[k + key_cont], auroc))
            key_cont += 4
        
        print(" ")
        print("------------------------- Ensemble Models results: ---------------------------")
        print("Classifiers' metrics, using a threshold of maximum 1% FPR (if available) and a minimum 50% TPR")
        for (i, em_chars) in enumerate(CHARACTERISTICS_EM):
            print(" ")
            metrics = distriblock_ensembles(train_meas, test_meas, val_meas, adv_meas, adv_val_meas, CHARACTERISTICS_EM[i])   
            print("Ensemble model using characteristics \"{}\":".format(em_chars))
            print("Accuracy: {:.2%} TP: {} FP: {} TN: {} FN: {} FPR: {:.2f} TPR: {:.2f} precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5], metrics[6], metrics[7], metrics[8], metrics[9]))

        print(" ")
        print("------------------------- Neural Network classifier results: ---------------------------")
        metrics = distriblock_NN(train_meas, test_meas, adv_meas, adv_val_meas, f"{orig_path}/NN-model.pth")   
        print("NN AUROC: {:.4f}".format(metrics[0]))
        print("Accuracy: {:.2%} TP: {} FP: {} TN: {} FN: {} FPR: {:.2f} TPR: {:.2f} precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(metrics[1], metrics[2], metrics[3], metrics[4], metrics[5], metrics[6], metrics[7], metrics[8], metrics[9], metrics[10]))
    else:
        sys.exit("-------------Error when Characteristics were calculated-------------")

