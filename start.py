import argparse

from create_model import create_model
from utils import save_model, parse_textfile

parser = argparse.ArgumentParser(
    description='Creates and trains a small Convolutional Neural Network to classify images of flowers',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='Number of times to train over data.'
)
parser.add_argument(
    '--train_batch_size',
    type=int,
    default=32,
    help='How many images to train on at a time.'
)
parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.01,
    help='Rate at which to train the model.'
)
parser.add_argument(
    '--save_hdf5',
    type=str,
    default='',
    help='Saves the model as an HDF5 file.'
)
parser.add_argument(
    '--save_history_plot',
    type=str,
    default='',
    help='Save a plot of the model accuracy and loss as the model is being trained as a PNG.'
)

args = parser.parse_args()
epochs = args.epochs
batch_size = args.train_batch_size
learning_rate = args.learning_rate
model_save_path = args.save_hdf5
history_plot_path = args.save_history_plot

# Print arguments
print('\nINPUT PARAMETERS')
print(f'Number of epochs to train for - {epochs}')
print(f'Number of samples to train on at a time - {batch_size}')
print(f'{learning_rate} will be the learning rate for the SGD algorithm')
if model_save_path:
    print(f'Will save model after training to - {model_save_path}')
if history_plot_path:
    print(f'Will save training plot to - {history_plot_path}')
print()

print('Parsing data text file\n')
# Returns a list with the images and a 1D numpy array with the classification number
(train_dataset, test_dataset) = parse_textfile()

print('Creating and Training Model\n')
model = create_model(train_dataset, test_dataset, batch_size, epochs, learning_rate, history_plot_path)
