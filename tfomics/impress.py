import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import matplotlib.cm as cm


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



def plot_filters(W, fig, num_cols=8, alphabet='ACGT', names=None, fontsize=12):
  """plot 1st layer convolutional filters"""

  num_filter, filter_len, A = W.shape
  num_rows = np.ceil(num_filter/num_cols).astype(int)

  fig.subplots_adjust(hspace=0.2, wspace=0.2)
  for n, w in enumerate(W):
    ax = fig.add_subplot(num_rows,num_cols,n+1)
    
    # Calculate sequence logo heights -- information
    I = np.log2(4) + np.sum(w * np.log2(w+1e-7), axis=1, keepdims=True)
    logo = I*w

    # Create DataFrame for logomaker
    counts_df = pd.DataFrame(data=0.0, columns=list(alphabet), index=list(range(filter_len)))
    for a in range(A):
      for l in range(filter_len):
        counts_df.iloc[l,a] = logo[l,a]

    logomaker.Logo(counts_df, ax=ax)
    ax = plt.gca()
    ax.set_ylim(0,2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    plt.xticks([])
    plt.yticks([])
    if names:
      plt.ylabel(names[n], fontsize=fontsize)


#--------------------------------------------------------------------------------
# pretty plots
#--------------------------------------------------------------------------------
  
def box_violin_plot(scores, cmap='tab10', ylabel=None, xlabel=None, title=None, fontsize=14):
  """ Plot box-violin plot to compare list of values within scores """

  # plot violin plot
  vplot = plt.violinplot(scores, showextrema=False);

  # set colors for biolin plot
  num_colors = len(scores)
  cmap = cm.ScalarMappable(cmap=cmap)
  color_mean = np.linspace(0, 1, num_colors)      
  for patch, color in zip(vplot['bodies'], cmap.to_rgba(color_mean)):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
        
  # plot box plot
  bplot = plt.boxplot(scores, notch=True, patch_artist=True, widths=0.2,    
                      medianprops=dict(color="red",linewidth=2), showfliers=False)

  # set colors for box plot
  for patch, color in zip(bplot['boxes'], cmap.to_rgba(color_mean)):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')

  # set up plot params
  ax = plt.gca();
  plt.setp(ax.get_yticklabels(), fontsize=fontsize)
  plt.setp(ax.get_xticklabels(), fontsize=fontsize)

  if ylabel is not None:
    plt.ylabel(ylabel, fontsize=fontsize);
  if xlabel is not None:
    plt.xticks(range(1,num_colors+1), xlabel, fontsize=fontsize, rotation=45, horizontalalignment="right");
  if title is not None:
    plt.title(title, fontsize=fontsize)

  return ax


def plot_history(history, names=['train','valid'], metrics=['loss'], ylabel=['Loss'], fontsize=12):
  """ Plot history from training"""
  num_metrics = len(metrics)

  fig = plt.figure(figsize=(num_metrics*5, 3))
  for i, metric in enumerate(metrics):
    ax = plt.subplot(1, num_metrics, i+1)
    for name in names:
      plt.plot(history[name+'_'+metric])
      plt.ylabel(ylabel[i], fontsize=fontsize)
      plt.xlabel('Epoch', fontsize=fontsize)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
      item.set_fontsize(fontsize)
  return fig



#--------------------------------------------------------------------------------
# utility functions for logomaker
#--------------------------------------------------------------------------------

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





