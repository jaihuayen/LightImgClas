import argparse

parser = argparse.ArgumentParser(description='PyTorch Lightning Image Classification')

# Hyperparameter options
parser.add_argument('--lr', default=1e-02, type=float, help='learning rate')

# Dataset options
parser.add_argument('--data', default=None, type=str, help='path to images dataset')
parser.add_argument('--train', default=None, type=str, help='path to training set csv file')
parser.add_argument('--val', default=None, type=str, help='path to validation set csv file')
parser.add_argument('--test', default=None, type=str, help='path to testing  set csv file')

# Dataloader options
parser.add_argument('--train_batch_size', default=None, type=int, help='training batch size')
parser.add_argument('--val_batch_size', default=None, type=int, help='validation batch size')
parser.add_argument('--test_batch_size', default=None, type=int, help='testing batch size')

# Enviornment options
parser.add_argument('--workers', default=0, type=int, help='number of data loader workers')

# Model options
parser.add_argument('--modelname', default='resnet50', type=str, help='model architecture')
parser.add_argument('--pretrain', action='store_true', help='set for transfer learning')
parser.add_argument('--num_classes', default=420, type=str, help='number of model outputs')
parser.add_argument('--num_epochs', default=100, type=str, help='number of model training epochs')
parser.add_argument('--train_all_layers', action='store_true', help='set for all layer learning')


# Other options
parser.add_argument('--c', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint')


def get_config():
    return parser.parse_args()
