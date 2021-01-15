import os 
import numpy as np
import tensorflow as tf



def function_batch(X, fun, batch_size=128, **kwargs):
  """ run a function in batches """

  dataset = tf.data.Dataset.from_tensor_slices(X)
  outputs = []
  for x in dataset.batch(batch_size):
    outputs.append(fun(x, **kwargs))
  return np.concatenate(outputs, axis=0)



def random_shuffle(x, num_samples=1):
  """ randomly shuffle sequences 
      assumes x shape is (N,L,A) """

  x_shuffle = []
  for i in range(num_samples):
    shuffle = np.random.permutation(x.shape[1])
    x_shuffle.append(x[0,shuffle,:])
  return np.array(x_shuffle)



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










  
 



