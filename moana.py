import pandas as pd
import numpy as np
from tensorflow import keras
import subprocess

# MOANA (MOtif ANAlysis)

#---------------------------------------------------------------------------------------
# Get position probability matrix of conv filters

def filter_activations(x_test, model, layer=3, window=20, threshold=0.5):
  """get alignment of filter activations for visualization"""

  # get feature maps of 1st convolutional layer after activation
  intermediate = keras.Model(inputs=model.inputs, outputs=model.layers[layer].output)
  fmap = intermediate.predict(x_test)
  N,L,A = x_test.shape

  # Set the left and right window sizes
  window_left = int(window/2)
  window_right = window - window_left

  W = []
  for filter_index in range(fmap.shape[-1]):

    # Find regions above threshold
    coords = np.where(fmap[:,:,filter_index] > np.max(fmap[:,:,filter_index])*threshold)
    x, y = coords

    # Sort score
    index = np.argsort(fmap[x,y,filter_index])[::-1]
    data_index = x[index].astype(int)
    pos_index = y[index].astype(int)

    # Make a sequence alignment centered about each activation (above threshold)
    seq_align = []
    for i in range(len(pos_index)):

      # Determine position of window about each filter activation
      start_window = pos_index[i] - window_left
      end_window = pos_index[i] + window_right

      # Check to make sure positions are valid
      if (start_window > 0) & (end_window < L):
        seq = x_test[data_index[i],start_window:end_window,:] 
        seq_align.append(seq)
    
    # Calculate position probability matrix
    if len(seq_align) > 0:
      W.append(np.mean(seq_align, axis=0))
    else:
      W.append(np.ones((window, A))/4)
  return np.array(W)


#---------------------------------------------------------------------------------------
# utilities to process filters

def clip_filters(W, threshold=0.5, pad=3):
  """clip uninformative parts of conv filters"""
  W_clipped = []
  for w in W:
    L,A = w.shape
    entropy = np.log2(4) + np.sum(w*np.log2(w+1e-7), axis=1)
    index = np.where(entropy > threshold)[0]
    if index.any():
      start = np.maximum(np.min(index)-pad, 0)
      end = np.minimum(np.max(index)+pad+1, L)
      W_clipped.append(w[start:end,:])
    else:
      W_clipped.append(w)

  return W_clipped


def meme_generate(W, output_file='meme.txt', prefix='filter'):
  """generate a meme file for a set of filters, W ∈ (N,L,A)"""

  # background frequency
  nt_freqs = [1./4 for i in range(4)]

  # open file for writing
  f = open(output_file, 'w')

  # print intro material
  f.write('MEME version 4\n')
  f.write('\n')
  f.write('ALPHABET= ACGT\n')
  f.write('\n')
  f.write('Background letter frequencies:\n')
  f.write('A %.4f C %.4f G %.4f T %.4f \n' % tuple(nt_freqs))
  f.write('\n')

  for j, pwm in enumerate(W):
    L, A = pwm.shape
    f.write('MOTIF %s%d \n' % (prefix, j))
    f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (L, L))
    for i in range(L):
      f.write('%.4f %.4f %.4f %.4f \n' % tuple(pwm[i,:]))
    f.write('\n')

  f.close()


def count_meme_entries(motif_path):
  """Count number of meme entries"""
  with open(motif_path, 'r') as f:
    counter = 0
    for line in f:
      if line[:6] == 'letter':
        counter += 1
  return counter


#---------------------------------------------------------------------------------------
# motif comparison

def tomtom(motif_path, jaspar_path, output_path, evalue=False, thresh=0.5, dist='pearson', png=None, tomtom_path='tomtom'):
  """ perform tomtom analysis """
  "dist: allr | ​ ed | ​ kullback | ​ pearson | ​ sandelin"
  cmd = [tomtom_path,'-thresh', str(thresh), '-dist', dist]
  if evalue:
    cmd.append('-evalue')  
  if png:
    cmd.append('-png')
  cmd.extend(['-oc', output_path, motif_path, jaspar_path])

  process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()
  return stdout, stderr


#---------------------------------------------------------------------------------------
# evaluation of tomtom motif comparison

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
  num_matches = len(np.unique(df['Query_ID'])) - 3.  # 3 is correction because of last 3 lines of comments in the tsv file (may change across tomtom versions)
  match_any = (num_matches - correction)/num_filters  # counts hits to any motif (not including Grembl)

  # match fraction to ground truth motifs
  match_index = np.where(filter_qvalue != 1.)[0]
  if any(match_index):
    match_fraction = len(match_index)/float(num_filters)
  else:
    match_fraction = 0.  

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




