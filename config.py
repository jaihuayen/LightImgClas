import argparse

parser = argparse.ArgumentParser(description='PyTorch Lightning Image Classification')

# Dataset options
parser.add_argument('--data', default=None, type=str, help='path to images dataset')
parser.add_argument('--train', default=None, type=str, help='path to training set csv file')
parser.add_argument('--val', default=None, type=str, help='path to validation set csv file')
parser.add_argument('--test', default=None, type=str, help='path to testing  set csv file')

# Dataloader options
parser.add_argument('--train_batch_size', default=None, type=str, help='training batch size')
parser.add_argument('--val_batch_size', default=None, type=str, help='validation batch size')
parser.add_argument('--test_batch_size', default=None, type=str, help='testing batch size')

# Enviornment options
parser.add_argument('--workers', default=0, type=int, help='number of data loader workers (default: 0)')

def get_config():
    return parser.parse_args()