import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker


def plot_attribution_map(saliency_df, ax=None, figsize=(20,1)):
  """plot an attribution map using logomaker"""

  logomaker.Logo(saliency_df, figsize=figsize, ax=ax)
  if ax is None:
    ax = plt.gca()
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.yaxis.set_ticks_position('none')
  ax.xaxis.set_ticks_position('none')
  plt.xticks([])
  plt.yticks([])



def plot_filters(W, fig, num_cols=8, alphabet='ACGT'):
  """plot 1st layer convolutional filters"""

  num_filter, filter_len, A = W.shape
  num_rows = np.ceil(num_filter/num_cols)

  fig.subplots_adjust(hspace=0.1, wspace=0.1)
  for n, w in enumerate(W):
    ax = fig.add_subplot(num_rows,num_cols,n+1)
    
    # Calculate sequence logo heights -- information
    I = np.log2(4) + np.sum(w * np.log2(w+1e-7), axis=1, keepdims=True)
    logo = I*w

    # Create DataFrame for logomaker
    counts_df = pd.DataFrame(data=0.0, columns=list(alphabet), index=list(range(num_filter)))
    for a in range(A):
      for l in range(filter_len):
        counts_df.iloc[l,a] = logo[l,a]
  return counts_df

  


def matrix_to_df(x, w, alphabet='ACGT'):
  """generate pandas dataframe for saliency plot
     based on grad x inputs """

  L, A = w.shape
  counts_df = pd.DataFrame(data=0.0, columns=list(alphabet), index=list(range(L)))
  for a in range(A):
      for l in range(L):
          counts_df.iloc[l,a] = w[l,a]
  return counts_df



def prob_to_info_df(w, alphabet='ACGT'):
  """generate pandas dataframe for saliency plot
     based on grad x inputs """


  # Calculate sequence logo heights -- information
  I = np.log2(4) + np.sum(w * np.log2(w+1e-7), axis=1, keepdims=True)
  logo = I*w

  L, A = logo.shape
  counts_df = pd.DataFrame(data=0.0, columns=list(alphabet), index=list(range(L)))
  for a in range(A):
      for l in range(L):
          counts_df.iloc[l,a] = logo[l,a]
  return counts_df


def grad_times_input_to_df(x, grad, alphabet='ACGT'):
  """generate pandas dataframe for saliency plot
     based on grad x inputs """

  x_index = np.argmax(np.squeeze(x), axis=1)
  grad = np.squeeze(grad)
  L, A = grad.shape

  seq = ''
  saliency = np.zeros((L))
  for i in range(L):
    seq += alphabet[x_index[i]]
    saliency[i] = grad[i,x_index[i]]

  # create saliency matrix
  saliency_df = logomaker.saliency_to_matrix(seq=seq, values=saliency)
  return saliency_df

  

def l2_norm_to_df(x, scores, alphabet='ACGT'):
  """generate pandas dataframe for saliency plot
     based on l2-norm of scores (i.e. mutagenesis) """

  x = np.squeeze(x)
  x_index = np.argmax(x, axis=1)
  scores = np.squeeze(scores)
  L, A = scores.shape

  # calculate l2-norm
  scores = np.sqrt(np.sum(scores**2, axis=1) + 1e-10)

  # create dataframe
  seq = ''
  saliency = np.zeros((L))
  for i in range(L):
    seq += alphabet[x_index[i]]
    saliency[i] = scores[i]

  # create saliency matrix
  saliency_df = logomaker.saliency_to_matrix(seq=seq, values=saliency)
  return saliency_df





