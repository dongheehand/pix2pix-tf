import argparse
from mode import *


def str2bool(v):
    return v.lower() in ('true')

parser = argparse.ArgumentParser()

parser.add_argument('--tr_data_path', type = str, default = './dataSet/facades/train')
parser.add_argument('--val_data_path', type = str, default = './dataSet/facades/val')
parser.add_argument('--transfer_type', type = str, default = 'A_to_B')
parser.add_argument('--data_ext', type = str, default = 'jpg')
parser.add_argument('--in_memory', type = str2bool, default = True)
parser.add_argument('--load_size', type = int, default = 286)
parser.add_argument('--crop_size', type = int, default = 256)
parser.add_argument('--random_jitter', type = str2bool, default = True)
parser.add_argument('--mirroring', type = str2bool, default = True)
parser.add_argument('--channel', type = int, default = 3)
parser.add_argument('--batch_size', type = int, default = 1)
parser.add_argument('--epoch', type = int, default = 200)
parser.add_argument('--mode', type = str, default = 'train')
parser.add_argument('--learning_rate', type = float, default = 0.0002)
parser.add_argument('--beta1', type = float, default = 0.5)
parser.add_argument('--beta2', type = float, default = 0.999)
parser.add_argument('--eps', type = float, default = 1e-12)
parser.add_argument('--pre_trained_model', type = str, default = './model/')
parser.add_argument('--model_name', type = str, default = 'facades')


args = parser.parse_args()

conf = {}
conf['tr_data_path'] = args.tr_data_path
conf['val_data_path'] = args.val_data_path
conf['transfer_type'] = args.transfer_type
conf['data_ext'] = args.data_ext
conf['in_memory'] = args.in_memory
conf['load_size'] = args.load_size
conf['crop_size'] = args.crop_size
conf['random_jitter'] = args.random_jitter
conf['mirroring'] = args.mirroring
conf['channel'] = args.channel
conf['batch_size'] = args.batch_size
conf['epoch'] = args.epoch
conf['mode'] = args.mode
conf['learning_rate'] = args.learning_rate
conf['beta1'] = args.beta1
conf['beta2'] = args.beta2
conf['eps'] = args.eps
conf['pre_trained_model'] = args.pre_trained_model
conf['model_name'] = args.model_name

if conf['mode'] == 'train':
    train(conf)
    
elif conf['mode'] == 'test':
    test(conf)

