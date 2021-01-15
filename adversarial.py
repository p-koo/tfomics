import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow import keras
from keras import backend as K
import tensorflow.compat.v1.keras.backend as K1
import utils, fit


#-------------------------------------------------------------------------------------------
# Tensorflow-based attack
#-------------------------------------------------------------------------------------------

@tf.function
def pgd_batch(model, input_var, x, y, num_steps, epsilon, adam_updates):
    x_pgd = tf.identity(x)       
    input_var.assign(x_pgd)
    for i in range(num_steps):    
        # setup gradient-based perturbations
        with tf.GradientTape() as tape:
            predictions = model(input_var)
            loss_value = model.loss(y, predictions)
            delta = tape.gradient(loss_value, input_var)
            
        # update perturbations
        x_pgd = adam_updates.update(x_pgd, delta)

        # project to l-infinity ball
        x_pgd = tf.clip_by_value(x_pgd, x-epsilon, x+epsilon) 
        
        # assign to input variable
        input_var.assign(x_pgd)
    return x_pgd


@tf.function
def fgsm_batch(model, input_var, x, y, epsilon):
    x_pgd = tf.identity(x)    

    input_var.assign(x_pgd)

    # setup gradient-based perturbations
    with tf.GradientTape() as tape:
        predictions = model(input_var)
        loss_value = model.loss(y, predictions)
        delta = tape.gradient(loss_value, input_var)
        
    return x + epsilon * tf.math.sign(delta)


def pgd_attack(model, x_test, y_test, num_steps, epsilon, batch_size=128):

    testset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    batch_dataset = testset.batch(batch_size)

    input_var = tf.Variable(x_test[:batch_size,:,:], shape=[None,200,4], dtype=tf.float32, trainable=True)   
    adam_updates = fit.AdamUpdates(x_test[:batch_size].shape)

    x_pgd_all = []
    for step, (x, y) in enumerate(batch_dataset):
        adam_updates.reset(x.shape)
        x_pgd = fit.pgd_batch(model, input_var, x, y, num_steps, epsilon, adam_updates)        
        x_pgd_all.append(x_pgd)
    return np.concatenate(x_pgd_all)



def fgsm_attack(model, x_test, y_test, epsilon, batch_size=128):

    testset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    batch_dataset = testset.batch(batch_size)

    input_var = tf.Variable(x_test[:batch_size,:,:], shape=[None,200,4], dtype=tf.float32, trainable=True)   

    x_pgd = []
    for step, (x, y) in enumerate(batch_dataset):
        x_pgd.append(fgsm_batch(model, input_var, x, y, epsilon)     )
    return np.concatenate(x_pgd)



#-------------------------------------------------------------------------------------------
# Keras-based attack
#-------------------------------------------------------------------------------------------

def pgd_linfinity_attack(model, x_train, y_train, epsilon=0.1, num_steps=10, noise=None, batch_size=100):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""

    # setup ops for fast gradient sign attack
    y_true = K.placeholder()
    loss = keras.losses.binary_crossentropy(y_true, model.output, from_logits=False)
    gradient = K1.gradients(loss, model.input)
    signed_grad = K.sign(gradient[0])

    # add Gaussian noise to inputs
    if noise:
        x = x_train + np.random.normal(scale=noise, size=x_train.shape)
    else:
        x = np.copy(x_train)

    # projected gradient descent on L-infinity ball
    sess = K1.get_session()
    placeholders = [model.inputs[0], y_true]
    v = 0
    m = 0
    beta1 = 0.9
    beta2 = 0.999
    learning_rate = 0.001
    for i in range(num_steps):
        inputs = [x, y_train]
        delta = utils.run_function_batch(sess, signed_grad, model, placeholders, inputs, batch_size)

        # update inputs with scale signed gradients and Adam
        m = beta1*m + (1-beta1)*delta
        v = beta2*v + (1-beta2)*(delta**2)
        x += learning_rate*m / (np.sqrt(v) + 1e-10)

        # project to l-infinity ball
        x = np.clip(x, x_train-epsilon, x_train+epsilon) 

    return x


def fast_gradient_sign_attack(model, x_train, y_train, epsilon=0.1, noise=None, batch_size=100):

    # setup ops for fast gradient sign attack
    y_true = K.placeholder()
    loss = keras.losses.binary_crossentropy(y_true, model.output, from_logits=False)
    gradient = K1.gradients(loss, model.input)
    signed_grad = K.sign(gradient[0])

    # setup feed dictionary arguments
    sess = K1.get_session()
    placeholders = [model.inputs[0], y_true]
    inputs = [x_train, y_train]

    # add Gaussian noise to inputs
    if noise:
        inputs[0] += np.random.normal(scale=noise, size=x_train.shape)

    # calculate fast gradient sign
    delta = utils.run_function_batch(sess, signed_grad, model, placeholders, inputs, batch_size)

    return x_train + delta*epsilon

