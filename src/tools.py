import pickle
import sys
import numpy as np


def load_meas_data(file_path, benign_flg):
    """
    Load a file containing the Characteristics.

    :param file_path: *.pickle file path.
    :param benign_flg: Set 1 to Benign data and 0 to Adversarial data.
    :return: A dictionary containing the Characteristics.
    """
    with open(file_path, "rb") as file:
        measurements = pickle.load(file)
    if "Entropy mean" in measurements.keys():
        total_length = len(measurements["Entropy mean"])
    else:
        sys.exit("------------- Error: Missing Characteristics -------------")
    if benign_flg:
        measurements["benign_flg"] = [1] * total_length
    else:
        measurements["benign_flg"] = [0] * total_length
    return measurements

def list_load_data(file_path, benign_flg):
    """
    Load a file containing the metrics.

    :param file_path: *.pickle file path.
    :param benign_flg: Set 1 to Benign data and 0 to Adversarial data.
    :return: A list containing the metrics.
    """
    with open(file_path, "rb") as file:
        measurements = pickle.load(file)
    total_length = len(measurements)
    data_gc = []
    if benign_flg:
        flag_list = [1] * total_length
    else:
        flag_list = [0] * total_length
    data_gc.append(measurements)
    data_gc.append(flag_list)
    return data_gc

def merge(dict_1, dict_2, key):
    """
    Merge two dictionaries based on specific keys.

    :param dict_1: Dictionary variable.
    :param dict_2: Dictionary variable.
    :param key: Keys to use during the merge.
    :return: A dictionary merged based on specific keys.
    """
    dict_all = {x: dict_1[x] + dict_2[x] for x in key}
    return dict_all

def largest_within_delta(X, delta):
    """
    Returns the first index where the ndarray exceeds delta. 

    :param X: ndarray representing the true positive rates.
    :param delta: TPR threshold.
    :return: First index exceeding delta.
    """
    best_idx = 0
    for (i,k) in enumerate(X):
        if k >= delta:
            return i
    return best_idx

def smaller_within_delta_indx(X, indx, delta):
    """
    Returns the nearest index where the ndarray falls below delta and starting from a specified position. 

    :param X: ndarray representing the false positive rates.
    :param indx: Starting specified position.
    :param delta: FPR threshold.
    :return: Nearest index below delta.
    """
    best_idx = indx
    min_indx = indx
    for (i,k) in enumerate(X):
        if i >= min_indx:
            if k <= delta:
                best_idx = i
    return best_idx

def perf_measure(y_actual, y_hat):
    """
    Compites model performance metrics such as FPR, TPR, precision, recall, and F1 score. 

    :param y_actual: ndarray representing the Actual truth label.
    :param y_hat: ndarray representing the Predicted label.
    :return: Model performance metrics: TP, FP, TN, and FN.
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return(TP, FP, TN, FN)

def dict_list(data_meas):
    """
    Create a list where each entry corresponds to 24 characteristics.

    :param data_meas: Dictionary containing data related to Characteristics.
    :return: Formatted list for neural network training.
    """
    # Get per each sample its corresponding 24 characteristics.
    total_length = len(data_meas["Entropy mean"])
    data_values = []
    for i in range(total_length):
        temp_list = []
        for list_of_values in data_meas.values():
            temp_list.append(list_of_values[i])
        data_values.append(temp_list)
    return data_values

def newWER(x, y):
    x = x.split()
    y = y.split()
    n = len(x)
    m = len(y)
    k = min(n, m)
    d = np.zeros((k + 1) * (k + 1), dtype = np.uint8).reshape(k + 1, k + 1)
    for i in range(k + 1):
        for j in range(k + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, k + 1):
        for j in range(1, k + 1):
            if (x[i - 1] == y[j - 1]):
                d[i][j] = d[i - 1][j - 1]
            else:
                S = d[i - 1][j - 1] + 1
                I = d[i][j - 1] + 1
                D = d[i - 1][j] + 1
                d[i][j] = min(S, I, D)
    return d[k][k] * 1.0 / k

def newCER(x, y):
    # Convert the list into a string without spaces
    x = x.replace(" ", "")
    y = y.replace(" ", "")
    n = len(x)
    m = len(y)
    k = min(n, m)
    d = np.zeros((k + 1) * (k + 1), dtype = np.uint8).reshape(k + 1, k + 1)

    for i in range(k + 1):
        for j in range(k + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, k + 1):
        for j in range(1, k + 1):
            if (x[i - 1] == y[j - 1]):
                d[i][j] = d[i - 1][j - 1]
            else:
                S = d[i - 1][j - 1] + 1
                I = d[i][j - 1] + 1
                D = d[i - 1][j] + 1
                d[i][j] = min(S, I, D)
	
    return d[k][k] * 1.0 / k

def word_error_rate(reference, hypothesis):
    """
    Credits: https://github.com/AI-secure/Characterizing-Audio-Adversarial-Examples-using-Temporal-Dependency
    Calculate Word-error-rate.

    :param reference: Reference text.
    :param hypothesis: hypothesis text.
    :return: WER value.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    len_ref = len(ref_words)
    len_hyp = len(hyp_words)
    min_len = min(len_ref, len_hyp)
    distance_matrix = np.zeros((min_len + 1) * (min_len + 1), dtype=np.uint8).reshape(min_len + 1, min_len + 1)
    
    for ref_idx in range(min_len + 1):
        for hyp_idx in range(min_len + 1):
            if ref_idx == 0:
                distance_matrix[0][hyp_idx] = hyp_idx
            elif hyp_idx == 0:
                distance_matrix[ref_idx][0] = ref_idx
    
    for ref_idx in range(1, min_len + 1):
        for hyp_idx in range(1, min_len + 1):
            if ref_words[ref_idx - 1] == hyp_words[hyp_idx - 1]:
                distance_matrix[ref_idx][hyp_idx] = distance_matrix[ref_idx - 1][hyp_idx - 1]
            else:
                substitution = distance_matrix[ref_idx - 1][hyp_idx - 1] + 1
                insertion = distance_matrix[ref_idx][hyp_idx - 1] + 1
                deletion = distance_matrix[ref_idx - 1][hyp_idx] + 1
                distance_matrix[ref_idx][hyp_idx] = min(substitution, insertion, deletion)
    
    return distance_matrix[min_len][min_len] * 1.0 / min_len

def character_error_rate(reference, hypothesis):
    """
    Credits: https://github.com/AI-secure/Characterizing-Audio-Adversarial-Examples-using-Temporal-Dependency
    Calculate character-error-rate.

    :param reference: Reference text.
    :param hypothesis: hypothesis text.
    :return: CER value.
    """
    ref_words = reference.replace(" ", "")
    hyp_words = hypothesis.replace(" ", "")
    len_ref = len(ref_words)
    len_hyp = len(hyp_words)
    min_len = min(len_ref, len_hyp)
    distance_matrix = np.zeros((min_len + 1) * (min_len + 1), dtype=np.uint8).reshape(min_len + 1, min_len + 1)
    
    for ref_idx in range(min_len + 1):
        for hyp_idx in range(min_len + 1):
            if ref_idx == 0:
                distance_matrix[0][hyp_idx] = hyp_idx
            elif hyp_idx == 0:
                distance_matrix[ref_idx][0] = ref_idx
    
    for ref_idx in range(1, min_len + 1):
        for hyp_idx in range(1, min_len + 1):
            if ref_words[ref_idx - 1] == hyp_words[hyp_idx - 1]:
                distance_matrix[ref_idx][hyp_idx] = distance_matrix[ref_idx - 1][hyp_idx - 1]
            else:
                substitution = distance_matrix[ref_idx - 1][hyp_idx - 1] + 1
                insertion = distance_matrix[ref_idx][hyp_idx - 1] + 1
                deletion = distance_matrix[ref_idx - 1][hyp_idx] + 1
                distance_matrix[ref_idx][hyp_idx] = min(substitution, insertion, deletion)
    
    return distance_matrix[min_len][min_len] * 1.0 / min_len