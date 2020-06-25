from torch import nn

class GANDALF(nn.Module):
	def __init__(self, input_dim, num_layers, hidden_size, activations, dropout_ratios):
		super(GANDALF, self).__init__()
		self.num_layers = num_layers

		if (num_layers > 1 and not isinstance(hidden_size, list)):
			hidden_size = [hidden_size] * num_layers
		if (num_layers > 1 and not isinstance(activations, list)):
			activations = [activations] * num_layers
		if (num_layers > 1 and not isinstance(dropout_ratios, list)):
			dropout_ratios = [dropout_ratios] * num_layers

		self.forward_layer_1 = FeedforwardBatchNormLayer(input_dim=input_dim, output_dim=hidden_size[0],
														 activation=activations[0], dropout_ratio=dropout_ratios[0])

		for i, (size, activation, dropout_ratio) in enumerate(zip(hidden_size[1:], activations[1:], dropout_ratios[1:]), 1):
			layer = FeedforwardBatchNormLayer(input_dim=hidden_size[i-1], output_dim=input_dim if size == 'XXX' else size,
											  activation=activation, dropout_ratio=dropout_ratio)
			setattr(self, f'forward_layer_{i+1}', layer)

	def forward(self, seq):
		for i in range(self.num_layers):
			seq = getattr(self, f'forward_layer_{i+1}')(seq)

		return seq


