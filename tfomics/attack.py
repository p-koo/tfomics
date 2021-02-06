
import tensorflow as tf


"""
@tf.function
def input_grad_batch(input_var, y, model, loss):
  with tf.GradientTape() as tape:
    predictions = model(input_var)
    loss_value = loss(y, predictions)
    return tape.gradient(loss_value, input_var) 
"""
@tf.function
def input_grad_batch(X, y, model, loss):
  """fast function to generate saliency maps"""
  if not tf.is_tensor(X):
    X = tf.Variable(X)

  with tf.GradientTape() as tape:
    tape.watch(X)
    predictions = model(X)
    loss_value = loss(y, predictions)
    return tape.gradient(loss_value, X) 

        

class PGDAttack():
  def __init__(self, shape, model, loss, learning_rate=0.1, epsilon=0.1, num_steps=10, grad_sign=True, decay=False):
    
    self.shape = [None].extend(list(shape[1:]))
    self.model = model
    self.loss = loss
    self.learning_rate = tf.Variable(learning_rate, trainable=False)
    self.epsilon = epsilon
    self.grad_sign = grad_sign
    self.num_steps = num_steps
    self.decay = decay

  def generate(self, x, y):
    x_pgd = tf.identity(x)
    for i in range(self.num_steps):

      delta = input_grad_batch(x_pgd, y, self.model, self.loss)

      # convert gradient to a sign (works better than pure gradients)
      if self.grad_sign:
        delta = tf.math.sign(delta)  

      # decay learning rate
      if self.decay:
        learning_rate = self.learning_rate/(i+10)
      else:
        learning_rate = self.learning_rate

      # update inputs   
      x_pgd += learning_rate*delta     
      
      # clip so as to project onto max perturbation of epsilon
      x_pgd = tf.clip_by_value(x_pgd, x-self.epsilon, x+self.epsilon) 
      #x = tf.clip_by_value(x, 0, 1) # ensure valid pixel range
    return x_pgd 



class FGSMAttack():
  def __init__(self, shape, model, loss, epsilon=0.1):
    
    self.shape = [None].extend(list(shape[1:]))
    self.model = model
    self.loss = loss
    self.epsilon = epsilon

  def generate(self, x, y):
    delta = input_grad_batch(x, y, self.model, self.loss)
    return x + self.epsilon*tf.math.sign(delta)



class NoiseAttack():
  def __init__(self, shape, mean=0.0, stddev=0.1):
    self.shape = shape
    self.mean = mean
    self.stddev = stddev

  def generate(self, x, y):
    return x + tf.random.normal(x.shape, mean=self.mean, stddev=self.stddev)
    


