import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def match_hits_to_ground_truth(file_path, motifs, num_filters=32):
    
  # get dataframe for tomtom results
  df = pd.read_csv(file_path, delimiter='\t')
  
  # loop through filters
  best_qvalues = np.ones(num_filters)
  best_match = np.zeros(num_filters)
  correction = 0  
  for name in np.unique(df['Query_ID'][:-3].to_numpy()):
    filter_index = int(name.split('r')[1])

    # get tomtom hits for filter
    subdf = df.loc[df['Query_ID'] == name]
    targets = subdf['Target_ID'].to_numpy()

    # loop through ground truth motifs
    for k, motif in enumerate(motifs): 

      # loop through variations of ground truth motif
      for motifid in motif: 

        # check if there is a match
        index = np.where((targets == motifid) ==  True)[0]
        if len(index) > 0:
          qvalue = subdf['q-value'].to_numpy()[index]

          # check to see if better motif hit, if so, update
          if best_qvalues[filter_index] > qvalue:
            best_qvalues[filter_index] = qvalue
            best_match[filter_index] = k 

    index = np.where((targets == 'MA0615.1') ==  True)[0]
    if len(index) > 0:
      if len(targets) == 1:
        correction += 1

  # get the minimum q-value for each motif
  num_motifs = len(motifs)
  min_qvalue = np.zeros(num_motifs)
  for i in range(num_motifs):
    index = np.where(best_match == i)[0]
    if len(index) > 0:
      min_qvalue[i] = np.min(best_qvalues[index])

  match_index = np.where(best_qvalues != 1)[0]
  if any(match_index):
    match_fraction = len(match_index)/float(num_filters)
  else:
    match_fraction = 0
  
  num_matches = len(np.unique(df['Query_ID']))-3
  match_any = (num_matches - correction)/num_filters

  return best_qvalues, best_match, min_qvalue, match_fraction, match_any



def interpretability_performance(X, score, X_model):

  score = np.sum(score, axis=2)
  pr_score = []
  roc_score = []
  for j, gs in enumerate(score):

    # calculate information of ground truth
    gt_info = np.log2(4) + np.sum(X_model[j]*np.log2(X_model[j]+1e-10),axis=1)

    # set label if information is greater than 0
    label = np.zeros(gt_info.shape)
    label[gt_info > 0.01] = 1

    # precision recall metric
    precision, recall, thresholds = precision_recall_curve(label, gs)
    pr_score.append(auc(recall, precision))

    # roc curve
    fpr, tpr, thresholds = roc_curve(label, gs)
    roc_score.append(auc(fpr, tpr))

  roc_score = np.array(roc_score)
  pr_score = np.array(pr_score)

  return roc_score, pr_score




  