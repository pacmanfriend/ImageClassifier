import numpy as np


# Activation function - ReLU
def relu(x):
    return np.maximum(0, x)


# Convolution operation
def convolution(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i + kernel_height, j:j + kernel_width] * kernel)

    return output


# Pooling operation
def max_pooling(image, pool_size):
    image_height, image_width = image.shape
    pool_height, pool_width = pool_size

    output_height = image_height // pool_height
    output_width = image_width // pool_width

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.max(image[i * pool_height:(i + 1) * pool_height, j * pool_width:(j + 1) * pool_width])

    return output


# Fully connected layer
def fully_connected(x, weights, biases):
    return np.dot(x, weights) + biases


# Softmax function
def softmax(x):
    exp_values = np.exp(x - np.max(x))
    return exp_values / np.sum(exp_values)


# Training data
image = np.random.rand(32, 32)  # Input image
label = 7  # Target class

# Convolutional layer
kernel = np.random.rand(3, 3)  # Convolutional kernel
conv_output = convolution(image, kernel)
conv_output = relu(conv_output)

# Pooling layer
pool_size = (2, 2)  # Pooling size
pool_output = max_pooling(conv_output, pool_size)

# Flatten the output
flatten_output = pool_output.flatten()

# Fully connected layer
weights = np.random.rand(flatten_output.shape[0], 100)  # Fully connected weights
biases = np.random.rand(100)  # Fully connected biases
fc_output = fully_connected(flatten_output, weights, biases)
fc_output = relu(fc_output)

# Softmax layer
output = softmax(fc_output)

# Calculate loss
loss = -np.log(output[label])

# Backpropagation
# Calculate gradient of softmax layer
gradient_softmax = output
gradient_softmax[label] -= 1

# Calculate gradient of fully connected layer
gradient_fc = gradient_softmax.dot(weights.T)
gradient_fc[fc_output <= 0] = 0

# Calculate gradient of pooling layer
gradient_pool = gradient_fc.reshape(pool_output.shape)

# Calculate gradient of convolutional layer
gradient_conv = np.zeros_like(conv_output)
kernel_height, kernel_width = kernel.shape

for i in range(gradient_pool.shape[0]):
    for j in range(gradient_pool.shape[1]):
        gradient_conv[i:i + kernel_height, j:j + kernel_width] += gradient_pool[i, j] * kernel

# Update kernel weights
learning_rate = 0.01
kernel -= learning_rate * convolution(image, np.rot90(gradient_conv, 2))

# Update fully connected weights and biases
weights -= learning_rate * np.outer(flatten_output, gradient_fc)
biases -= learning_rate * gradient_fc
