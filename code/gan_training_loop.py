from argparse import ArgumentParser
import csv
import logging
import os
import sys

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
from sacred.observers import SqlObserver
from sacred.observers import TelegramObserver
from sacred.observers import TinyDbObserver
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from torch import nn
from wolkenatlas.embedding import Embedding
import numpy as np
import torch

from .pytorch_models import GANDALF
from . import io
from . import wittgenstein

PACKAGE_PATH = os.path.dirname(__file__)
PROJECT_PATH = PACKAGE_PATH

parser = ArgumentParser()
parser.add_argument('-i', '--input-file', type=str, help='input file')
parser.add_argument('-ip', '--input-path', type=str, required=True, help='path to input file')
parser.add_argument('-i2', '--input-file-2', type=str, help='input file 2')
parser.add_argument('-ip2', '--input-path-2', type=str, help='path to input file 2')
parser.add_argument('-i3', '--input-file-3', type=str, help='input file 3')
parser.add_argument('-ip3', '--input-path-3', type=str, help='path to input file 3')
parser.add_argument('-op', '--output-path', type=str, help='path to output file')
parser.add_argument('-op2', '--output-path-2', type=str, help='path to output file 2')
parser.add_argument('-o2', '--output-file-2', type=str, help='output file 2')
parser.add_argument('-op3', '--output-path-3', type=str, help='path to output file 3')
parser.add_argument('-o3', '--output-file-3', type=str, help='output file 3')
parser.add_argument('-op4', '--output-path-4', type=str, help='path to output file 4')
parser.add_argument('-cn', '--config-name', type=str, required=True, help='name of config')
parser.add_argument('-ef', '--experiment-file', type=str, required=True)
parser.add_argument('-s', '--store-scores', action='store_true', help='Store individual scores in file.')
parser.add_argument('-sp', '--score-path', type=str, help='path to store the score file.')
parser.add_argument('-eid', '--experiment-id', type=int, default=-1, help='experiment id to use.')
parser.add_argument('-obs', '--sacred-observers', nargs='+', type=str, default=['mongo', 'telegram', 'sqlite'],
					help='mongo observers to add')
parser.add_argument('-ll', '--log-level', type=str, default='INFO', help='logging level', choices=['CRITICAL', 'FATAL',
																								   'ERROR', 'WARNING',
																								   'WARN', 'INFO',
																								   'DEBUG'])
parser.add_argument('-em', '--evaluation-mode', type=str, default='validation', help='evaluation mode')
parser.add_argument('-mp', '--min-performance', type=float, default=0.68, help='minimum performance of model for serialising it')
parser.add_argument('-kp', '--kill-performance', type=float, default=0.45, help='stop process automatically if performance stays below'
																				'given level after some burn-in period')
parser.add_argument('-bi', '--burn-in', type=int, default=5, help='number of epochs of patience before the kill-performance option'
																  'becomes active')
parser.add_argument('-egt', '--experiment-id-greater-than', type=int, default=-1, help='skip first n experiments')

ex = Experiment('GANDALF')
OBSERVERS = {
	'mongo': MongoObserver.create(db_name='GANDALF'),
	'telegram': TelegramObserver.from_config(os.path.join(PROJECT_PATH, 'resources/sacred/telegram.json')),
	'sqlite': SqlObserver.create(
		'sqlite:///{}'.format(os.path.join(PACKAGE_PATH, 'resources', 'sacred', 'GANDALF.sqlite'))),
	'tinydb': TinyDbObserver.create(os.path.join(PACKAGE_PATH, 'resources', 'sacred', 'GANDALF.tinydb')),
	'file': FileStorageObserver.create(
		os.path.join(PACKAGE_PATH, 'resources', 'sacred', 'GANDALF.fs'))
}


@ex.config
def config():
	config_name = ''
	exp_name = ''
	input_file = ''
	vector_file = ''
	vector_model_type = ''
	gen_hidden_size = []
	disc_hidden_size = []
	gen_activations = []
	disc_activations = []
	gen_dropout_ratios = []
	disc_dropout_ratios = []
	num_epochs = -1
	optimiser_class = ''
	learning_rate = -0.
	experiment_id = -1
	beta_1 = 0.
	beta_2 = 0.
	batch_size = 0
	output_file = ''
	min_avg_acc = 0.
	model_out_path = ''
	random_seed = -1
	noun_file = ''
	lambada = 0.
	augment_gen_loss = False



def load_embeddings_for_nouns(noun_file, emb):
	logging.info(f'Loading nouns from {noun_file}...')
	data = io.load_structured_resource(noun_file)
	words = []
	w = []

	for n in data:
		if n in emb:
			words.append(n)
			w.append(emb[n])
	logging.info(f'Collected {len(words)} nouns!')

	X = torch.FloatTensor(normalize(np.array(w), axis=0))

	return X


def lr_check(generator, epoch, rnd, X, exp_id):
	logging.info(f'\t[Epoch={epoch}] - Evaluating real vs fake data with a linear classifier...')
	X_noise = torch.FloatTensor(rnd.normal(loc=0., scale=1., size=X.shape))

	F = generator(X_noise).detach().numpy()
	R = X.detach().numpy()

	t = np.hstack((np.ones((R.shape[0])), np.zeros((R.shape[0]))))
	D = np.vstack((R, F))
	kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2906)

	accs = []
	for idx, (train_idx, test_idx) in enumerate(kf.split(D, t)):
		X_train, X_test = D[train_idx], D[test_idx]
		y_train, y_test = t[train_idx], t[test_idx]

		lr = LogisticRegression(solver='liblinear')
		lr.fit(X_train, y_train)

		y_pred = lr.predict(X_test)

		acc = accuracy_score(y_test, y_pred)
		accs.append(acc)
		logging.debug(f'\t\t[Epoch={epoch}] - Accuracy at split={idx}: {acc}')
	logging.info(f'\t[Experiment ID={exp_id}; Epoch={epoch}] - Accuracy: {np.average(accs)} (+/- {np.std(accs)})')

	return accs


def train_discriminator(generator, discriminator, rnd, X, loss_function, d_optimiser):
	discriminator.zero_grad()
	y_real_pred = discriminator(X)

	# Create label noise (soft labels + swap some of them)
	idx = rnd.uniform(0, 1, y_real_pred.shape)
	idx = np.argwhere(idx < 0.03)

	ones = np.ones(y_real_pred.shape) + rnd.uniform(-0.1, 0.1)
	ones[idx] = 0

	zeros = np.zeros(y_real_pred.shape) + rnd.uniform(0, 0.2)
	zeros[idx] = 1

	ones = torch.from_numpy(ones).float()
	zeros = torch.from_numpy(zeros).float()

	loss_real = loss_function(y_real_pred, ones)

	X_noise = torch.FloatTensor(rnd.normal(loc=0., scale=1., size=X.shape))
	X_fake = generator(X_noise)

	y_fake_pred = discriminator(X_fake)
	loss_fake = loss_function(y_fake_pred, zeros)

	loss = loss_real + loss_fake
	loss.backward()

	d_optimiser.step()

	return loss


def train_generator(generator, discriminator, loss_function, g_optimiser, rnd, shape, lr_acc, lambada, augment_gen_loss):
	X_noise = torch.FloatTensor(rnd.normal(loc=0., scale=1., size=shape))

	generator.zero_grad()
	X_fake = generator(X_noise)

	y_fake = discriminator(X_fake)

	ones = torch.ones_like(y_fake)

	loss = loss_function(y_fake, ones)

	if (augment_gen_loss and lr_acc > 0):
		loss += (lr_acc - 0.5) * lambada
	loss.backward()

	g_optimiser.step()

	return loss, X_fake


@ex.main
def run(vector_file, vector_model_type, gen_hidden_size, disc_hidden_size, gen_activations, disc_activations,
		gen_dropout_ratios, disc_dropout_ratios, num_epochs, optimiser_class, learning_rate, experiment_id,
		beta_1, beta_2, batch_size, output_file, random_seed, model_out_path, min_avg_acc, augment_gen_loss,
		lambada, noun_file):
	logging.info(f'[Experiment ID={experiment_id}] - Loading vectors of type={vector_model_type} from {vector_file}...')
	emb = Embedding(model_file=vector_file)
	logging.info('Embeddings loaded!')

	# Make sure random seeds are consistent & reproducible
	if (random_seed != -1):
		rnd = np.random.RandomState(seed=random_seed)
		torch.manual_seed(random_seed)

	X_all = load_embeddings_for_nouns(noun_file=noun_file, emb=emb)

	if (batch_size > 0):
		batches = torch.chunk(X_all, int(X_all.shape[0] / batch_size))
	else:
		batches = [X_all]

	logging.info('Initialising Generator & Discriminator...')
	# Different models for positive and negative antecedents and consequents
	generator = GANDALF(input_dim=X_all.shape[1], num_layers=len(gen_hidden_size),
						activations=gen_activations, dropout_ratios=gen_dropout_ratios,
						hidden_size=gen_hidden_size)
	discriminator = GANDALF(input_dim=X_all.shape[1], num_layers=len(disc_hidden_size),
							activations=disc_activations, dropout_ratios=disc_dropout_ratios,
							hidden_size=disc_hidden_size)
	loss_function = nn.BCELoss()
	g_optimiser = wittgenstein.create_instance(optimiser_class, params=generator.parameters(), lr=learning_rate,
											   betas=(beta_1, beta_2))
	d_optimiser = wittgenstein.create_instance(optimiser_class, params=discriminator.parameters(), lr=learning_rate,
											   betas=(beta_1, beta_2))
	logging.info('Networks initialised!')

	logging.info(f'Running networks for {num_epochs}...')
	disc_losses = []
	gen_losses = []
	avg_accs = []
	best_avg_acc = 1.
	curr_avg_acc = -1
	for epoch in range(num_epochs):
		for X in batches:
			# Train discriminator
			loss_discriminator = train_discriminator(generator=generator, discriminator=discriminator, rnd=rnd, X=X,
													 loss_function=loss_function, d_optimiser=d_optimiser)

			# Train generator
			loss_generator, X_fake = train_generator(generator=generator, discriminator=discriminator, rnd=rnd,
													 shape=X.shape, g_optimiser=g_optimiser, loss_function=loss_function,
													 augment_gen_loss=augment_gen_loss, lr_acc=curr_avg_acc,
													 lambada=lambada)

			disc_losses.append(loss_discriminator.item())
			gen_losses.append(loss_generator.item())
		logging.debug(f'Epoch={epoch}/{num_epochs};\n'
					  f'Discriminator Loss={loss_discriminator.item()}\n'
					  f'Generator Loss={loss_generator.item()}\n')

		if (epoch % 10 == 0): # Checkpoint every 10 epochs or so if the model is worth it
			accs = lr_check(generator=generator, epoch=epoch, rnd=rnd, X=X_all, exp_id=experiment_id)
			avg_acc = np.average(accs)
			avg_accs.append(avg_acc)
			curr_avg_acc = avg_acc

			if (avg_acc < best_avg_acc and avg_acc < min_avg_acc):
				best_avg_acc = avg_acc
				if (not model_out_path.endswith('none')):
					logging.info(f'Storing models at {model_out_path}...')
					io.save_pytorch_model(generator, os.path.join(model_out_path, f'generator_exp_id-{experiment_id}_epoch-{epoch}.pytorch'))
					io.save_pytorch_model(discriminator, os.path.join(model_out_path, f'discriminator_exp_id-{experiment_id}_epoch-{epoch}.pytorch'))
					logging.info('Models stored!')

			logging.info(f'Epoch={epoch}/{num_epochs};\n'
						 f'Discriminator Loss={loss_discriminator.item()}\n'
						 f'Generator Loss={loss_generator.item()}\n'
						 f'Average Accuracy={np.average(accs)}\n')

	logging.info('Creating final fake data...')
	accs = lr_check(generator=generator, epoch=num_epochs, rnd=rnd, X=X_all, exp_id=experiment_id)
	avg_accs.append(np.average(accs))
	logging.info(f'Experiment ID={experiment_id} - Final Average Accuracy={np.average(accs)}')

	logging.info(f'Storing result file at {output_file}...')
	results = {
		'discriminator_loss': disc_losses,
		'generator_loss': gen_losses,
		'avg_accuracies': avg_accs
	}
	io.save_structured_resource(results, output_file)
	logging.info('Finished!')

	return np.average(accs)


if (__name__ == '__main__'):
	args = parser.parse_args()

	timestamped_foldername = path_utils.timestamped_foldername()
	log_path = os.path.join(path_utils.get_log_path(), timestamped_foldername)
	if (not os.path.exists(log_path)):
		os.makedirs(log_path)

	log_formatter = logging.Formatter(fmt='%(asctime)s: %(levelname)s - %(message)s', datefmt='[%d/%m/%Y %H:%M:%S %p]')
	root_logger = logging.getLogger()
	root_logger.setLevel(getattr(logging, args.log_level))

	file_handler = logging.FileHandler(
		os.path.join(log_path, 'debug_gan_2_{}.log'.format(args.config_name)))
	file_handler.setFormatter(log_formatter)
	root_logger.addHandler(file_handler)

	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(log_formatter)
	root_logger.addHandler(console_handler)

	if (args.output_path is not None and not os.path.exists(args.output_path)):
		os.makedirs(args.output_path)

	if (args.output_path_2 is not None and args.output_path_2 != 'none' and not os.path.exists(args.output_path_2)):
		os.makedirs(args.output_path_2)

	if (args.output_path_3 is not None and not os.path.exists(args.output_path_3)):
		os.makedirs(args.output_path_3)

	if (args.output_path_4 is not None and not os.path.exists(args.output_path_4)):
		os.makedirs(args.output_path_4)

	for obs in args.sacred_observers:
		ex.observers.append(OBSERVERS[obs])
	ex.logger = root_logger

	# Load experiment id file
	with open(os.path.join(PACKAGE_PATH, 'resources', args.experiment_file), 'r') as csv_file:
		csv_reader = csv.reader(csv_file)
		experiments = []

		for line in csv_reader:
			experiments.append(line)

		if (args.experiment_id > 0):
			experiments = [experiments[args.experiment_id - 1]]

	for experiment_id, (vector_file, vector_model_type, gen_hidden_size, disc_hidden_size, gen_activations,
						disc_activations, gen_dropout_ratios,  disc_dropout_ratios, num_epochs, optimiser_class,
						learning_rate, beta_1, beta_2, batch_size, output_file, random_seed, noun_file,
						augment_gen_loss, lambada) \
			in enumerate(experiments):

		if (experiment_id > args.experiment_id_greater_than):
			logging.info(f'Running Experiment with id={experiment_id}...')
			config_dict = {
				'vector_file': os.path.join(args.input_path, vector_file),
				'vector_model_type': vector_model_type,
				'gen_hidden_size': list(map(lambda x: int(x), gen_hidden_size.split('-'))),
				'disc_hidden_size': list(map(lambda x: int(x), disc_hidden_size.split('-'))),
				'gen_activations': gen_activations.split('-'),
				'disc_activations': disc_activations.split('-'),
				'gen_dropout_ratios': list(map(lambda x: float(x), gen_dropout_ratios.split('-'))),
				'disc_dropout_ratios': list(map(lambda x: float(x), disc_dropout_ratios.split('-'))),
				'num_epochs': int(num_epochs),
				'optimiser_class': optimiser_class,
				'learning_rate': float(learning_rate),
				'experiment_id': experiment_id,
				'config_name': args.config_name,
				'exp_name': args.config_name,
				'beta_1': float(beta_1),
				'beta_2': float(beta_2),
				'batch_size': int(batch_size),
				'output_file': os.path.join(args.output_path, output_file.format(experiment_id)),
				'min_avg_acc': args.min_performance,
				'model_out_path': args.output_path_2,
				'random_seed': int(random_seed),
				'noun_file' os.path.join(args.input_path_2, noun_file)
				'augment_gen_loss': augment_gen_loss=='True',
				'lambada': float(lambada)
			}
			ex.run(config_updates=config_dict)
		else:
			logging.info(f'Skipping experiment with id={experiment_id} (skipping first {args.experiment_id_greater_than}) experiments!')
