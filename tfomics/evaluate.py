import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from . import moana 


def interpretability_performance(scores, x_model, threshold=0.01):
  """ Compare attribution scores to ground truth (e.g. x_model).
      scores --> (N,L)
      x_model --> (N,L,A)
  """

  pr_score = []
  roc_score = []
  for j, score in enumerate(scores):

    # calculate information of ground truth
    gt_info = np.log2(4) + np.sum(x_model[j]*np.log2(x_model[j]+1e-10),axis=1)

    # set label if information is greater than 0
    label = np.zeros(gt_info.shape)
    label[gt_info > threshold] = 1

    # (don't evaluate over low info content motif positions)
    index = np.where((gt_info > threshold) | (gt_info == np.min(gt_info)))[0]

    # precision recall metric
    precision, recall, thresholds = precision_recall_curve(label[index], score[index])
    pr_score.append(auc(recall, precision))

    # roc curve
    fpr, tpr, thresholds = roc_curve(label[index], score[index])
    roc_score.append(auc(fpr, tpr))

  roc_score = np.array(roc_score)
  pr_score = np.array(pr_score)

  return roc_score, pr_score


def signal_noise_stats(scores, x_model, top_k=10, threshold=0.01):
  """averate saliency score at signals and average noise level. Signal and 
     noise are determined by information content of sequence model (x_model)"""

  signal = []
  noise_mean = []
  noise_max = []
  noise_topk = []
  for j, score in enumerate(scores):

    # calculate information of ground truth
    gt_info = np.log2(4) + np.sum(x_model[j]*np.log2(x_model[j]+1e-10),axis=1)

    # (don't evaluate over low info content motif positions)
    index = np.where(gt_info > threshold)[0]
    signal.append(np.mean(score[index]))
    
    # evaluate noise levels
    index = np.where((score > 0) & (gt_info == np.min(gt_info)))[0]
    noise_max.append(np.max(score[index]))
    noise_mean.append(np.mean(score[index]))

    sort_score = np.sort(score[index])[::-1]
    noise_topk.append(np.mean(sort_score[:top_k]))

  return np.array(signal), np.array(noise_max), np.array(noise_mean), np.array(noise_topk)



def motif_comparison_synthetic_dataset(file_path, num_filters=32):
  """ Compares tomtom analysis for filters trained on synthetic multitask classification.
      Works with Tomtom version 5.1.0 
 
  inputs:
      - file_path: .tsv file output from tomtom analysis
      - num_filters: number of filters in conv layer (needed to normalize -- tomtom doesn't always give results for every filter)

  outputs:
      - match_fraction: fraction of hits to ground truth motifs
      - match_any: fraction of hits to any motif in JASPAR (except Gremb1)
      - filter_match: the motif of the best hit (to a ground truth motif)
      - filter_qvalue: the q-value of the best hit to a ground truth motif (1.0 means no hit)
      - motif_qvalue: for each ground truth motif, gives the best qvalue hit
      - motif_counts for each ground truth motif, gives number of filter hits
  """

  arid3 = ['MA0151.1', 'MA0601.1', 'PB0001.1']
  cebpb = ['MA0466.1', 'MA0466.2']
  fosl1 = ['MA0477.1']
  gabpa = ['MA0062.1', 'MA0062.2']
  mafk = ['MA0496.1', 'MA0496.2']
  max1 = ['MA0058.1', 'MA0058.2', 'MA0058.3']
  mef2a = ['MA0052.1', 'MA0052.2', 'MA0052.3']
  nfyb = ['MA0502.1', 'MA0060.1', 'MA0060.2']
  sp1 = ['MA0079.1', 'MA0079.2', 'MA0079.3']
  srf = ['MA0083.1', 'MA0083.2', 'MA0083.3']
  stat1 = ['MA0137.1', 'MA0137.2', 'MA0137.3', 'MA0660.1', 'MA0773.1']
  yy1 = ['MA0095.1', 'MA0095.2']

  motifs = [arid3, cebpb, fosl1, gabpa, mafk, max1, mef2a, nfyb, sp1, srf, stat1, yy1]
  motif_names = ['Arid3', 'CEBPB', 'FOSL1', 'GABPA', 'MAFK', 'MAX', 'MEF2A', 'NFYB', 'SP1', 'SRF', 'STAT1', 'YY1']
  match_fraction, match_any, filter_match, filter_qvalue, min_qvalue, num_counts = moana.match_hits_to_ground_truth(file_path, motifs, motif_names, num_filters)

  return match_fraction, match_any, filter_match, filter_qvalue, min_qvalue, num_counts

