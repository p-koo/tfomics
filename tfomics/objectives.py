import tensorflow as tf

def binary_cross_entropy(targets, predictions, weights=None, keepdims=False, axis=None):
  """ Standard binary cross entroyp loss. """
  predictions = tf.clip_by_value(predictions,1e-7,1-1e-7)
  if weights is not None:
    return tf.math.reduce_mean(weights*(targets*tf.math.log(predictions) + (1-targets)*tf.math.log(1-predictions)), keepdims=keepdims, axis=axis)
  else:
    return tf.math.reduce_mean(targets*tf.math.log(predictions) + (1-targets)*tf.math.log(1-predictions), keepdims=keepdims, axis=axis)


def categorical_cross_entropy(targets, predictions, weights=None, keepdims=False, axis=None):
  """ Standard categorical cross entroyp loss. """
  predictions = tf.clip_by_value(predictions,1e-7,1-1e-7)
  if weights is not None:
    return tf.math.reduce_mean(weights*targets*tf.math.log(predictions), keepdims=keepdims, axis=axis)
  else:
    return tf.math.reduce_mean(targets*tf.math.log(predictions), keepdims=keepdims, axis=axis)
        

def squared_error(targets, predictions, weights=None, keepdims=False, axis=None):
  """ Standard Squared error loss. """
  if weights is not None:
    return tf.math.reduce_sum(weights*tf.math.square(targets - predictions), keepdims=keepdims, axis=axis)
  else:
    return tf.math.reduce_sum(tf.math.square(targets - predictions), keepdims=keepdims, axis=axis)


def kld_gaussian(z_mu, z_logvar, axis=None):
  """KL Divergence between Gaussian (z_mu, z_logvar) and standard normal gaussian. Useful for VAEs.""" 
  z_sigma = tf.math.sqrt(tf.math.exp(z_logvar))
  return 0.5*tf.math.reduce_sum(1 + 2*tf.math.log(z_sigma) - tf.math.square(Z_mu) - tf.math.exp(2*tf.math.log(z_sigma)), axis=axis)


def kld_softmax(z, axis=None):
  """ KL Divergence between Categorical (z) and uniform distribution. Useful for VAEs."""
  num_class = tf.shape(z)[-1]
  z = tf.clip_by_value(z,1e-7,1-1e-7)
  return tf.math.reduce_sum( z*(tf.math.log(z) - tf.math.log(1.0/num_class)), axis=axis)

