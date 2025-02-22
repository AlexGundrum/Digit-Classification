#Alex G
import tensorflow as tf
from tensorflow import keras
import math
import numpy as np
from keras.datasets import mnist

class DenseLayer:
    
    def __init__(self, input_size, output_size, activation):
        #Setting up shape of weights tensor
        weights_shape = (input_size, output_size)
        weights_initial_value = tf.random.uniform(weights_shape, minval= 0, maxval = 1e-1)
        self.w = tf.Variable(weights_initial_value)
        #Setting up shape of biases tensor
        biases_shape = (output_size,)
        biases_initial = tf.zeros(biases_shape)
        self.biases = tf.Variable(biases_initial)
        
        self.activation = activation

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.w) + self.biases)

    @property
    def weights(self):
        return [self.w, self.biases]
    

class SequentialLayer:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, inputs):
        '''
        Takes the inputs, and passes that through the first layer. Then, the output of the first layer is the input of the second.
        This is repeated until the last layer outputs the last output.
        '''
        
        x = inputs
        
        for layer in self.layers:
            x = layer(x)
        
        return x

    @property
    def weights(self):
        weights_list = []
        
        for layer in self.layers:
            weights_list += layer.weights
        
        return weights_list
    

class BatchGenerator:
    def __init__(self, images, labels, batch_size = 128):
        assert len(images) == len(labels), "Inconsistent number of images and labels passed!"
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.number_of_batches = math.ceil(len(images) / batch_size)
        self.index = 0

    
    def next(self):
        '''
        Returns the next batch of images, labels
        '''
        next_images = self.images[self.index: self.index + self.batch_size]
        next_labels = self.labels[self.index: self.index + self.batch_size]
        self.index += self.batch_size
        return [next_images, next_labels]
    

def one_step_through_model(model, images_batch, labels_batch):
    '''
    Given a model, it will take the batch of images and labels, do a forward pass through the model, 
    and use TensorFlow's GradientTape to know how to update the weights of the model.
    '''
    with tf.GradientTape() as tape:

        predictions = model(images_batch)
        per_sample_loss = tf.keras.losses.sparse_categorical_crossentropy(labels_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_loss)
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss



'''
In practice, "update_weights" would be replaced with 
optimizer = optimizers.SGD(learning_rate)

def update_weights(gradients, weights):
optimizer.apply(zip(gradients, weights))

but for practice this suffices.
'''
learning_rate = 1e-3
#Taking the gradient and moving the weights by the learning rate in the minimizing direction.
def update_weights(gradients, weights):
    for gradient, weight in zip(gradients, weights):
        weight.assign_sub(gradient * learning_rate)



def fit_model(model, images, labels, epochs, batch_size = 128):
    for epoch_number in range(epochs):
        print(f"Epoch number: {epoch_number}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.number_of_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_step_through_model(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"Loss at batch: {batch_counter} is :{loss}")



'''
This specific model has input size of 28*28 so that each pixel's value can be entered into the model.
The first layer uses Rectified Linear Unit (ReLU), which takes each output, and does the following:
    output = max(output, 0)
The second layer takes the output of the first layer, and turns outputs the probabilities of each digit. 
This is why the second layer's input size is equal to the output size of the first layer, and the second layer has output size of 10 (one for each digit 0-9).
The softmax activation turns the real numbers its given into a probabalistic distribution corresponding to the probability of each output (summing to 0).

'''
model = SequentialLayer([
    DenseLayer(input_size = 28*28, output_size=512, activation=tf.nn.relu),
    DenseLayer(input_size = 512, output_size = 10, activation=tf.nn.softmax)
])

assert len(model.weights) == 4, "Model was initialized incorrectly!"


(train_images, train_labels), (test_images, test_labels) = mnist.load_data() #unpacking the data from the dataset

#Transforming the data to a form that can be inputted and understood by the model
train_images = train_images.reshape((60_000, 28 * 28)) 
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10_000, 28 * 28))
test_images = test_images.astype("float32") / 255

fit_model(model, train_images, train_labels, 1000, 128)





#evaluation of the model:
predictions = model(test_images)
predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"accuracy: {matches.mean() : .2f}")



