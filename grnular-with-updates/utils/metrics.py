import numpy as np
import pdb
import warnings
import sklearn
from sklearn import metrics

def get_auc(y, scores):
    y = np.array(y).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(y, scores)
    return auroc, aupr

def report_metrics(G_true, G, beta=1, PRINT=False):
    G_true = G_true.real
    G =G.real
    if PRINT:
        print('Check report metrics: ', G_true, G)
    # G_true and G are numpy arrays
    # convert all non-zeros in G to 1
    d = G.shape[-1]

    # changing to 1/0 for TP and FP calculations
    G_binary = np.where(G!=0, 1, 0)
    G_true_binary = np.where(G_true!=0, 1, 0)
    # extract the upper diagonal matrix
    indices_triu = np.triu_indices(d, 1)
    edges_true = G_true_binary[indices_triu] #np.triu(G_true_binary, 1)
    edges_pred = G_binary[indices_triu] #np.triu(G_binary, 1)
    # Getting AUROC value
    edges_pred_auc = G[indices_triu] #np.triu(G_true_binary, 1)
    auroc, auprc = get_auc(edges_true, np.absolute(edges_pred_auc))
    # Now, we have the edge array for comparison
    # true pos = pred is 1 and true is 1
    TP = np.sum(edges_true * edges_pred) # true_pos
    # False pos = pred is 1 and true is 0
    mismatches = np.logical_xor(edges_true, edges_pred)
    FP = np.sum(mismatches * edges_pred)
    # Find all mismatches with Xor and then just select the ones with pred as 1 
    # P = Number of pred edges : nnz_pred 
    P = np.sum(edges_pred)
    # T = Number of True edges :  nnz_true
    T = np.sum(edges_true)
    # F = Number of non-edges in true graph
    F = len(edges_true) - T
    # SHD = total number of mismatches
    SHD = np.sum(mismatches)
    
    # FDR = False discovery rate
    # with warnings.catch_warnings(record=True) as w:
    #     FDR = FP/P
    #     if len(w) > 0:
    #         pdb.set_trace()

    FDR = FP/P
    # TPR = True positive rate
    TPR = TP/T
    # FPR = False positive rate
    FPR = FP/F
    # False negative = pred is 0 and true is 1
    FN = np.sum(mismatches * edges_true)
    # F beta score
    num = (1+beta**2)*TP
    den = ((1+beta**2)*TP + beta**2 * FN + FP)
    F_beta = num/den
    # precision 
    
    precision = TP/(TP+FP)
    # recall 
    recall = TP/(TP+FN) 
#    print('FDR, TPR, FPR, SHD, nnz_true, nnz_pred, F1, auc')
    return FDR, TPR, FPR, SHD, T, P, precision, recall, F_beta, auprc, auroc 

def main():
    a_pred = np.array([[0.74, 0.02, 0.01, 0, 0], [0.02, 1.25, 0,0,0], [0.01,0,0.79,0,0], [0,0,0,0.81,0], [0,0,0,0,0.78]])
    # sign match check
    a_pred2 = np.array([[1.33, 0.32, 0,0,0], [0.32,1.33,0.02,0,0.08], [0, 0.02,1.33,0,0], [0,0,0,1.33,0], [0,0.08,0,0,1.33]])
  
    a_true = np.array([[1.33, 0.32, 0,0,0], [0.32,1.33,-0.02,0,-0.08], [0, -0.02,1.33,0,0], [0,0,0,1.33,0], [0,-0.08,0,0,1.33]])
    print(a_pred, a_true)
    print(report_metrics(a_true, a_pred))
    print(report_metrics(a_true, a_pred2))
    print(report_metrics(a_true, a_true))

if __name__=="__main__":
    main()

