import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def match_hits_to_ground_truth(file_path, motifs, motif_names=None, num_filters=32):
  """ works with Tomtom version 5.1.0 
  inputs:
      - file_path: .tsv file output from tomtom analysis
      - motifs: list of list of JASPAR ids
      - motif_names: name of motifs in the list
      - num_filters: number of filters in conv layer (needed to normalize -- tomtom doesn't always give results for every filter)

  outputs:
      - match_fraction: fraction of hits to ground truth motifs
      - match_any: fraction of hits to any motif in JASPAR (except Gremb1)
      - filter_match: the motif of the best hit (to a ground truth motif)
      - filter_qvalue: the q-value of the best hit to a ground truth motif (1.0 means no hit)
      - motif_qvalue: for each ground truth motif, gives the best qvalue hit
      - motif_counts for each ground truth motif, gives number of filter hits
  """

  # add a zero for indexing no hits
  motifs = motifs.copy()
  motif_names = motif_names.copy()
  motifs.insert(0, [''])
  motif_names.insert(0, '')

  # get dataframe for tomtom results
  df = pd.read_csv(file_path, delimiter='\t')

  # loop through filters
  filter_qvalue = np.ones(num_filters)
  best_match = np.zeros(num_filters).astype(int)
  correction = 0  
  for name in np.unique(df['Query_ID'][:-3].to_numpy()):
    filter_index = int(name.split('r')[1])

    # get tomtom hits for filter
    subdf = df.loc[df['Query_ID'] == name]
    targets = subdf['Target_ID'].to_numpy()

    # loop through ground truth motifs
    for k, motif in enumerate(motifs): 

      # loop through variations of ground truth motif
      for id in motif: 

        # check if there is a match
        index = np.where((targets == id) ==  True)[0]
        if len(index) > 0:
          qvalue = subdf['q-value'].to_numpy()[index]

          # check to see if better motif hit, if so, update
          if filter_qvalue[filter_index] > qvalue:
            filter_qvalue[filter_index] = qvalue
            best_match[filter_index] = k 

    # dont' count hits to Gmeb1 (because too many)
    index = np.where((targets == 'MA0615.1') ==  True)[0]
    if len(index) > 0:
      if len(targets) == 1:
        correction += 1

  # get names of best match motifs
  filter_match = [motif_names[i] for i in best_match]

  # get hits to any motif
  num_matches = len(np.unique(df['Query_ID'])) - 3  # 3 is correction because of last 3 lines of comments in the tsv file (may change across tomtom versions)
  match_any = (num_matches - correction)/num_filters  # counts hits to any motif (not including Grembl)

  # match fraction to ground truth motifs
  match_index = np.where(filter_qvalue != 1.)[0]
  if any(match_index):
    match_fraction = len(match_index)/float(num_filters)
  else:
    match_fraction = 0  

  # get the number of hits and minimum q-value for each motif
  num_motifs = len(motifs) - 1
  motif_qvalue = np.zeros(num_motifs)
  motif_counts = np.zeros(num_motifs)
  for i in range(num_motifs):
    index = np.where(best_match == i+1)[0]
    if len(index) > 0:
      motif_qvalue[i] = np.min(filter_qvalue[index])
      motif_counts[i] = len(index)

  return match_fraction, match_any, filter_match, filter_qvalue, motif_qvalue, motif_counts


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

    # precision recall metric
    precision, recall, thresholds = precision_recall_curve(label, score)
    pr_score.append(auc(recall, precision))

    # roc curve
    fpr, tpr, thresholds = roc_curve(label, score)
    roc_score.append(auc(fpr, tpr))

  roc_score = np.array(roc_score)
  pr_score = np.array(pr_score)

  return roc_score, pr_score


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
  motif_names = ['Arid3', 'CEBPB', 'FOSL1', 'GABPA', 'MAFK', 'MAX', 'MEF2A', 'NFYB', 'AP1', 'SRF', 'STAT1', 'YY1']
  match_fraction, match_any, filter_match, filter_qvalue, min_qvalue, num_counts = match_hits_to_ground_truth(file_path, motifs, motif_names, num_filters)

  return match_fraction, match_any, filter_match, filter_qvalue, min_qvalue, num_counts
