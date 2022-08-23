# -*- coding: utf-8 -*-


from RREA.runner import Runner
import argparse
from divea.neural_rrea import RREAModule
from divea.util import Config

parser = argparse.ArgumentParser()
parser.add_argument('--layer_num', type=int, default=2)
parser.add_argument('--max_train_epoch', type=int, default=500)
parser.add_argument('--training_mode', type=str, default="supervised")
parser.add_argument('--data_dir', type=str)
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir

conf = Config()
conf.data_dir = data_dir
conf.output_dir = output_dir
conf.max_train_epoch = args.max_train_epoch
conf.initial_training = args.training_mode
conf.second_device = "cuda:1"
conf.gcn_layer_num = args.layer_num

module = RREAModule(conf)
module.train_model()
module.evaluate()




