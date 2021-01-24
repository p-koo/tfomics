import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from scipy import stats



def evaluate(label, pred, metrics, verbose=True):      
  metric_vals = {}
  if 'acc' in metrics:
    metric_vals['acc'] = accuracy(label, pred)
  if 'auroc' in metrics:
    metric_vals['auroc'] = auroc(label, pred)
  if 'aupr' in metrics:
    metric_vals['aupr'] = aupr(label, pred)
  if 'rsquare' in metrics:
    metric_vals['rsquare'] = rsquare(label, pred)
  if 'corr' in metrics:
    metric_vals['corr'] = pearsonr(label, pred)
  if verbose:  
    for metric_name in metrics:
        print("%s:\t%.5f+/-%.5f"%(metric_name, np.mean(metric_vals[metric_name]), 
                                  np.std(metric_vals[metric_name])))        
  return metric_vals


def auroc(label, prediction):
  """Area under the ROC curve and ROC curves. 
     Input shapes are (N,C) where N is the # of data 
    and C is the # of classes. """
  num_labels = label.shape[1]
  auroc_score = np.zeros((num_labels))
  for i in range(num_labels):
    auroc_score[i] = roc_auc_score(label[:,i], prediction[:,i])
  return auroc_score


def aupr(label, prediction):
  """Area under the PR curve and PR curves. 
     Input shapes are (N,C) where N is the # of data 
    and C is the # of classes. """
  num_labels = label.shape[1]
  aupr_score = np.zeros((num_labels))
  for i in range(num_labels):
    aupr_score[i]= average_precision_score(label[:,i], prediction[:,i])  
  return aupr_score

  
def accuracy(label, prediction):
  """Binary accuracy. Input shapes are (N,C) 
     where N is the # of data and C is the # of classes. """
  num_labels = label.shape[1]
  acc_score = np.zeros((num_labels))
  for i in range(num_labels):
    acc_score[i] = accuracy_score(label[:,i], np.round(prediction[:,i]))
  return acc_score


def pearsonr(label, prediction):
  """Pearson correlation. Input shapes are (N,C) 
     where N is the # of data and C is the # of classes. """
  num_labels = label.shape[1]
  corr = []
  for i in range(num_labels):
    corr.append(stats.pearsonr(label[:,i], prediction[:,i])[0])		
  return corr


def rsquare(label, prediction):
  """R-squared of a linear fit. Input shapes are (N,C) 
     where N is the # of data and C is the # of classes. """
  num_labels = label.shape[1]
  metric = []
  slope = []
  for i in range(num_labels):
    y = label[:,i]
    X = prediction[:,i]
    m = np.dot(X,y)/np.dot(X, X)
    resid = y - m*X; 
    ym = y - np.mean(y); 
    rsqr2 = 1 - np.dot(resid.T,resid)/ np.dot(ym.T, ym);
    metric.append(rsqr2)
    slope.append(m)
  return metric, slope



