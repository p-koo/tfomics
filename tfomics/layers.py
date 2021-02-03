import tensorflow as tf
from tensorflow import keras


#-----------------------------------------------------------------------------
# Reverse complement layers
#-----------------------------------------------------------------------------

class RevCompSplit(keras.layers.Layer):
  """
  Split inputs into forward and reverse complements -- assumes alphabet is ACGT
  """
  def __init__(self, *args, **kwargs):
    super(RevCompSplit, self).__init__(*args, **kwargs)

  def call(self, inputs):
    return inputs, inputs[:,::-1,::-1]


class RevCompConv1D(keras.layers.Conv1D):
  """
  Implement forward and reverse-complement filter convolutions
  for 1D signals. It takes as input either a single input or two inputs 
  (where the second input is the reverse complement scan). If a single input, 
  this performs both forward and reverse complement scans and either merges it 
  (if concat=True) or returns a separate scan for forward and reverse comp. 
  """
  def __init__(self, *args, concat=False, **kwargs):
    super(RevCompConv1D, self).__init__(*args, **kwargs)
    self.concat = concat


  def call(self, inputs, inputs2=None):

    if inputs2 is not None:
      # create rc_kernels
      rc_kernel = self.kernel[::-1,::-1,:]

      # convolution 1D
      outputs = self._convolution_op(inputs, self.kernel)
      rc_outputs = self._convolution_op(inputs2, rc_kernel)

    else:
      # create rc_kernels
      rc_kernel = tf.concat([self.kernel, self.kernel[::-1,::-1,:]], axis=-1)

      # convolution 1D
      outputs = self._convolution_op(inputs, rc_kernel)

      # unstack to forward and reverse strands
      outputs = tf.unstack(outputs, axis=2)
      rc_outputs = tf.stack(outputs[self.filters:], axis=2)
      outputs = tf.stack(outputs[:self.filters], axis=2)

    # add bias
    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)
      rc_outputs = tf.nn.bias_add(rc_outputs, self.bias)

    # add activations
    if self.activation is not None:
      outputs = self.activation(outputs)
      rc_outputs = self.activation(rc_outputs)

    if self.concat:
      return tf.concat([outputs, rc_outputs], axis=-1)
    else:
      return outputs, rc_outputs

      

class RevCompMeanPool(keras.layers.Layer):
  """merge forward and reverse complement scans via mean pooling"""
  def __init__(self, **kwargs):
    super(RevCompMeanPool, self).__init__(**kwargs)

  def call(self, inputs, inputs2=None, reverse=True):
    if inputs2 is None:
      num_filters = inputs.get_shape()[2]//2
      fwd = inputs[:,:,:num_filters]
      rev = inputs[:,:,num_filters:]
    else:
      fwd = inputs
      rev = inputs2

    if reverse:
      rev = rev[:,::-1,:]
    return 0.5*(fwd + rev)


class RevCompMaxPool(keras.layers.Layer):
  """merge forward and reverse complement scans via max pooling"""
  def __init__(self, **kwargs):
    super(RevCompMaxPool, self).__init__(**kwargs)

  def call(self, inputs, inputs2=None, reverse=True):
    if inputs2 is None:
      num_filters = inputs.get_shape()[2]//2
      fwd = inputs[:,:,:num_filters]
      rev = inputs[:,:,num_filters:]
    else:
      fwd = inputs
      rev = inputs2

    if reverse:
      rev = rev[:,::-1,:]
    return tf.maximum(fwd, rev)


#-----------------------------------------------------------------------------
# Multi-head attention
#-----------------------------------------------------------------------------


class MultiHeadAttention(keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = keras.layers.Dense(d_model)
    self.wk = keras.layers.Dense(d_model)
    self.wv = keras.layers.Dense(d_model)
    
    self.dense = keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size, seq_len):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q):
    batch_size = tf.shape(q)[0]
    seq_len = tf.constant(q.shape[1])

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size, seq_len)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size, seq_len)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size, seq_len)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, seq_len, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights



@tf.function
def scaled_dot_product_attention(q, k, v):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights



#-----------------------------------------------------------------------------
# Sampling Layers
#-----------------------------------------------------------------------------


class GaussianSampleLayer(keras.layers.Layer):
  """
  Samples independent Gaussians using reparameterization trick 
  """

  def __init__(self, mean=0.0, stddev=0.1, **kwargs):
    super(GaussianSampleLayer, self).__init__(**kwargs)
    self.mean = mean
    self.stddev = stddev

  def call(self, input_mu, input_logvar):
    sigma = tf.math.sqrt(tf.math.exp(incoming_logvar) + 1e-7)
    z = tf.random_normal(shape=tf.shape(incoming_mu), mean=self.mean, stddev=self.stddev, dtype=input_mu.dtype)
    return input_mu + tf.multiply(sigma, z)
    
    

class CategoricalSampleLayer(keras.layers.Layer):
  """
  Samples independent Gaussians using reparameterization trick 
  """

  def __init__(self, **kwargs):
    super(CategoricalSampleLayer, self).__init__(**kwargs)

  def call(self, inputs, temperature, axis=1, hard=False):
    return gumbel_softmax(logits, temperature, axis, hard)
   



def gumbel_softmax_sample(logits, temperature, axis=None): 
  """ Draw a sample from the Gumbel-Softmax distribution"""

  def sample_gumbel(shape, eps=1e-20): 
    """Sample from Gumbel(0, 1)"""
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)

  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax(y/temperature, axis=axis)


def gumbel_softmax(logits, temperature, axis=1, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature, axis)
  if hard:
    # k = tf.shape(logits)[-1]
    # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.math.equal(y, tf.math.reduce_max(y, axis=axis, keep_dims=True)), y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y




#-----------------------------------------------------------------------------
# Other Layers
#-----------------------------------------------------------------------------











