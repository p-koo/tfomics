import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker


def make_directory(path, foldername, verbose=1):
  """make a directory"""

  if not os.path.isdir(path):
    os.mkdir(path)
    print("making directory: " + path)

  outdir = os.path.join(path, foldername)
  if not os.path.isdir(outdir):
    os.mkdir(outdir)
    print("making directory: " + outdir)
  return outdir



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


