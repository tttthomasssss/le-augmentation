from torch import nn
import torch

ACTIVATIONS = {
	'tanh': nn.Tanh,
	'sigmoid': nn.Sigmoid,
	'relu': nn.ReLU,
	'prelu': nn.PReLU,
	'selu': nn.SELU,
	'celu': nn.CELU,
	'elu': nn.ELU,
	'leaky_relu': nn.LeakyReLU,
	'softsign': nn.Softsign,
	'tanhshrink': nn.Tanhshrink,
	'softshrink': nn.Softshrink
}
