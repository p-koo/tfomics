import numpy as np
import tensorflow as tf
from tensorflow import keras

class Explainer():
  """wrapper class for attribution maps"""

  def __init__(self, model, class_index=None, func=tf.math.reduce_mean):

    self.model = model
    self.class_index = class_index
    self.func = func

  def saliency_maps(self, X, batch_size=128):
    
    return function_batch(X, saliency_map, batch_size, model=self.model, 
                          class_index=self.class_index, func=self.func) 

  def smoothgrad(self, X, num_samples=50, mean=0.0, stddev=0.1):
    
    return function_batch(X, smoothgrad, batch_size=1, model=self.model, 
                           num_samples=num_samples, mean=mean, stddev=stddev,
                           class_index=self.class_index, func=self.func) 


  def integrated_grad(self, X, baseline_type='random', num_steps=25):

    scores = []
    for x in X:
      x = np.expand_dims(x, axis=0)
      baseline = self.set_baseline(x, baseline_type, num_samples=1)
      intgrad_scores = integrated_grad(x, model=self.model, baseline=baseline,
                           num_steps=num_steps, class_index=self.class_index, func=self.func)
      scores.append(intgrad_scores)
    return np.concatenate(scores, axis=0)


  def expected_integrated_grad(self, X, num_baseline=25, baseline_type='random', num_steps=25):
    
    scores = []
    for x in X:
      x = np.expand_dims(x, axis=0)
      baselines = self.set_baseline(x, baseline_type, num_samples=num_baseline)
      intgrad_scores = expected_integrated_grad(x, model=self.model, baselines=baselines,
                           num_steps=num_steps, class_index=self.class_index, func=self.func)
      scores.append(intgrad_scores)
    return np.concatenate(scores, axis=0)


  def mutagenesis(self, X, class_index=None):
    scores = []
    for x in X:
      x = np.expand_dims(x, axis=0)
      scores.append(mutagenesis(x, self.model, class_index))
    return np.concatenate(scores, axis=0)


  def set_baseline(self, x, baseline, num_samples):
    if baseline == 'random':
      baseline = random_shuffle(x, num_samples)
    else:
      baseline = np.zeros((x.shape))
    return baseline


#------------------------------------------------------------------------------

@tf.function
def saliency_map(X, model, class_index=None, func=tf.math.reduce_mean):
  """fast function to generate saliency maps"""
  if not tf.is_tensor(X):
    X = tf.Variable(X)

  with tf.GradientTape() as tape:
    tape.watch(X)
    if class_index is not None:
      outputs = model(X)[:, class_index]
    else:
      outputs = func(model(X))
  return tape.gradient(outputs, X)


#------------------------------------------------------------------------------

def smoothgrad(x, model, num_samples=50, mean=0.0, stddev=0.1, 
               class_index=None, func=tf.math.reduce_mean):

  _,L,A = x.shape
  x_noise = tf.tile(x, (num_samples,1,1)) + tf.random.normal((num_samples,L,A), mean, stddev)
  grad = saliency_map(x_noise, model, class_index=class_index, func=func)
  return tf.reduce_mean(grad, axis=0, keepdims=True)


#------------------------------------------------------------------------------

def integrated_grad(x, model, baseline, num_steps=25, 
                         class_index=None, func=tf.math.reduce_mean):

  def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients
  
  def interpolate_data(baseline, x, steps):
    steps_x = steps[:, tf.newaxis, tf.newaxis]   
    delta = x - baseline
    x = baseline +  steps_x * delta
    return x

  steps = tf.linspace(start=0.0, stop=1.0, num=num_steps+1)
  x_interp = interpolate_data(baseline, x, steps)
  grad = saliency_map(x_interp, model, class_index=class_index, func=func)
  avg_grad = integral_approximation(grad)
  return avg_grad * (x - baseline)  


#------------------------------------------------------------------------------

def expected_integrated_grad(x, model, baselines, num_steps=25,
                             class_index=None, func=tf.math.reduce_mean):
  """average integrated gradients across different backgrounds"""

  grads = []
  for baseline in baselines:
    grads.append(integrated_grad(x, model, baseline, num_steps=num_steps, 
                                 class_index=class_index, func=tf.math.reduce_mean))
  return np.mean(np.array(grads), axis=0)


#------------------------------------------------------------------------------

def mutagenesis(x, model, class_index=None):
  """ in silico mutagenesis analysis for a given sequence"""

  def generate_mutagenesis(x):
    _,L,A = x.shape 
    x_mut = []
    for l in range(L):
      for a in range(A):
        x_new = np.copy(x)
        x_new[0,l,:] = 0
        x_new[0,l,a] = 1
        x_mut.append(x_new)
    return np.concatenate(x_mut, axis=0)

  def reconstruct_map(predictions):
    _,L,A = x.shape 
    
    mut_score = np.zeros((1,L,A))
    k = 0
    for l in range(L):
      for a in range(A):
        mut_score[0,l,a] = predictions[k]
        k += 1
    return mut_score

  def get_score(x, model, class_index):
    score = model.predict(x)
    if class_index == None:
      score = np.sqrt(np.sum(score**2, axis=-1, keepdims=True))
    else:
      score = score[:,class_index]
    return score

  # generate mutagenized sequences
  x_mut = generate_mutagenesis(x)
  
  # get baseline wildtype score
  wt_score = get_score(x, model, class_index)
  predictions = get_score(x_mut, model, class_index)

  # reshape mutagenesis predictiosn
  mut_score = reconstruct_map(predictions)

  return mut_score - wt_score


#------------------------------------------------------------------------------

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


#------------------------------------------------------------------------------
# Useful functions
#------------------------------------------------------------------------------


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
