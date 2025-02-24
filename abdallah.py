import random
import math

def tanh(x):
    return math.tanh(x)
def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))
def neural_network(inputs, weights1, weights2, b1, b2):
    hidden_layer_input = [dot_product(inputs, w) + b1 for w in weights1]
    hidden_layer_output = [tanh(x) for x in hidden_layer_input]

    output_layer_input = dot_product(hidden_layer_output, weights2) + b2
    output = tanh(output_layer_input)

    return output

inputs = [1.0, 0.5, -0.5]

weights1 = [[random.uniform(-0.5, 0.5) for _ in range(3)] for _ in range(4)]
weights2 = [random.uniform(-0.5, 0.5) for _ in range(4)]

b1 = 0.5
b2 = 0.7

output = neural_network(inputs, weights1, weights2, b1, b2)

print("Output of the network:", output)
