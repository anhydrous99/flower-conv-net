import argparse

from create_model import create_model
from utils import save_model, parse_textfile, load_images, split

parser = argparse.ArgumentParser(
    description='Creates and trains a small Convolutional Neural Network to classify images of flowers',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--data_path',
    type=str,
    default='FlowerData/6k_img_map.txt',
    help='Path to text file containing image paths and classifications.'
)
parser.add_argument(
    '--epochs',
    type=int,
    default=50,
    help='Number of times to train over data.'
)
parser.add_argument(
    '--disable_augmentation',
    help='Disables data augmentation',
    action='store_true'
)
parser.add_argument(
    '--train_batch_size',
    type=int,
    default=32,
    help='How many images to train on at a time.'
)
parser.add_argument(
    '--percentage_validation',
    type=int,
    default=10,
    help='Percentage of data to use as validation'
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
data_path = args.data_path
epochs = args.epochs
use_augmentation = not args.disable_augmentation
batch_size = args.train_batch_size
percent_valid = args.percentage_validation
model_save_path = args.save_hdf5
history_plot_path = args.save_history_plot

# Print arguments
print('\nINPUT PARAMETERS')
print(f'Path of data text file - {data_path}')
print(f'Number of epochs to train for - {epochs}')
print(f'Number of samples to train on at a time - {batch_size}')
print(f'{percent_valid}% of data will be used for validation')
if use_augmentation:
    print('Data augmentation will be used')
if model_save_path:
    print(f'Will save model after training to - {model_save_path}')
if history_plot_path:
    print(f'Will save training plot to - {history_plot_path}')
print()

print('Parsing data text file\n')
# Returns a list with the images and a 1D numpy array with the classification number
(image_list, classifications) = parse_textfile(data_path)

print('Loading images\n')
# Returns images in path list as a 4D numpy array with the following format (#images, height, width, channels)
image_array = load_images(image_list)

# Partitions data for validation and training
(x_train, y_train, x_test, y_test) = split(image_array, classifications, percent_valid)

print('Creating and Training Model\n')
model = create_model(x_train, y_train, x_test, y_test, batch_size, epochs, history_plot_path, use_augmentation)

# Saves the model to Kera's
if model_save_path:
    print('Saving model\n')
    save_model(model, model_save_path)
