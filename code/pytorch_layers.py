from torch import nn

from . import util


class FeedforwardBatchNormLayer(nn.Module):
	def __init__(self, input_dim, output_dim, activation, dropout_ratio):
		super(FeedforwardBatchNormLayer, self).__init__()
		if (activation is None or activation == 'none'):
			self.feedforward = nn.Linear(input_dim, output_dim)
		else:
			self.feedforward = nn.Sequential(
				nn.Linear(input_dim, output_dim),
				nn.BatchNorm1d(output_dim),
				util.ACTIVATIONS[activation](),
				nn.Dropout(p=dropout_ratio)
			)

	def forward(self, sequence):
		return self.feedforward(sequence)
