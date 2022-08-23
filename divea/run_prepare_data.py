# -*- coding: utf-8 -*-
#
# prepare data for rrea

from divea.data2 import UniformData, convert_uniform_to_rrea, convert_uniform_to_openea
import argparse
from divea.util import seed_everything


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--kgids', type=str)
parser.add_argument('--divide', action='store_true', default=False)
parser.add_argument('--train_percent', type=float, required=False)
parser.add_argument('--rrea', action='store_true', default=False)
parser.add_argument('--openea', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1011)
args = parser.parse_args()

seed_everything(args.seed)

kgids = args.kgids.split(",")

uni_data = UniformData(args.data_dir, kgids)

if args.divide:
    uni_data.divide_train_test(args.train_percent)

# if args.rrea:
#     convert_uniform_to_rrea(args.data_dir, kgids)
# if args.openea:
#     convert_uniform_to_openea(args.data_dir, kgids)






