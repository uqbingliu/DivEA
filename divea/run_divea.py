# -*- coding: utf-8 -*-
import torch

from divea.components_base import Client
from divea.neural_rrea import RREAModule
from divea.neural_gcnalign import GCNAlignModule
from divea.util import Config
import os
from divea.components_base import Server
import argparse
from divea.util import seed_everything
from divea.ctx_builder_onclient import CtxBuilderV2
from divea.framework import ParallelEAFramework
from divea.components import DivG1Metis, DivG1Random
from divea.components import CounterpartDiscovery
from divea.data2 import UniformData
from divea.ctx_builder_onclient import CtxBuilderV1


parser = argparse.ArgumentParser()

## data, output
parser.add_argument('--data_dir', type=str)
parser.add_argument('--kgids', type=str, help="separate two ids with comma. e.g. `fr,en`")
parser.add_argument('--output_dir', type=str)
# subtask
parser.add_argument('--subtask_num', type=int, required=True)
parser.add_argument('--subtask_size', type=str, default="1.0", help="specify size with int value; or specify ratio to average partition size with float value")  #
parser.add_argument('--ctx_g1_percent', type=float, default=0.5, help="percentage of first context graph size")
parser.add_argument('--ctx_g2_conn_percent', type=float, default=0.0, help="percentage of connecting entities in the second context graph")
# module configuration
parser.add_argument('--ctx_builder', type=str, default="v2", choices=["v1", "v2"], help="v1: build ctx on client; v2: build ctx on server.")
parser.add_argument('--div_g1', type=str, default="metis", choices=["metis", "random"], help="method of partitioning first graph; for detailed analysis")  # metis, random
parser.add_argument('--counterdisc_ablation', type=str, default="full", choices=["full", "locality"], help="for ablation study of counterpart discovery")
# hyper-parameters, see paper for the meanining of these hyper-parameters
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=2.0)
parser.add_argument('--topK', type=int, default=10)
parser.add_argument('--max_iteration', type=int, default=5)
# ea model
parser.add_argument('--ea_model', type=str, default="rrea", choices=["rrea", "gcn-align"], help="EA model")
parser.add_argument('--layer_num', type=int, default=2, help="number of GCN layers")
parser.add_argument('--max_train_epoch', type=int, default=50, help="max epoch of training EA model")
# device, running env
parser.add_argument('--gpu_ids', type=str, default="0", help="visible GPU devices")
parser.add_argument('--py_exe_fn', type=str)
# others
parser.add_argument('--seed', type=int, default=1011, help="random seed")


args = parser.parse_args()
data_dir = args.data_dir
kgids = args.kgids.split(",")
out_dir = args.output_dir

seed_everything(args.seed)

data = UniformData(data_dir, kgids)

subtask_size = eval(args.subtask_size)
if isinstance(subtask_size, float):
    total_kg_size = len(data.kg1_entities) + len(data.kg2_entities)
    subtask_size = int(subtask_size * total_kg_size / args.subtask_num)
ctx_g1_size = int(subtask_size * args.ctx_g1_percent)
ctx_g2_size = subtask_size - ctx_g1_size

if args.ctx_builder == "v1":
    ctx_builder = CtxBuilderV1(data, data_dir, kgids, out_dir, args.subtask_num, args.layer_num, gamma=args.gamma,
                                 ctx_g1_size=ctx_g1_size, ctx_g2_size=ctx_g2_size, ctx_g2_conn_percent=args.ctx_g2_conn_percent, torch_devices=[f"cuda:{didx}" for didx in range(len(args.gpu_ids.split(",")))])
else:
    ctx_builder = CtxBuilderV2(data, data_dir, kgids, out_dir, args.subtask_num, args.layer_num, gamma=args.gamma,
                               ctx_g1_size=ctx_g1_size, ctx_g2_size=ctx_g2_size, ctx_g2_conn_percent=args.ctx_g2_conn_percent, torch_devices=[f"cuda:{didx}" for didx in range(len(args.gpu_ids.split(",")))])

if args.div_g1 == "metis":
    g1_partitioner = DivG1Metis(data, data_dir, kgids, args.subtask_num)
elif args.div_g1 == "random":
    g1_partitioner = DivG1Random(data, data_dir, kgids, args.subtask_num)
else:
    raise Exception("unknown g1 partition method")


count_discover = CounterpartDiscovery(data, data_dir, kgids, args.subtask_num, ctx_g2_size=ctx_g2_size,
                                      max_hop_k=2*args.layer_num, out_dir=out_dir, alpha=args.alpha,
                                      beta=args.beta, topK=args.topK, ablation=args.counterdisc_ablation
                                      )


server = Server(data_dir, kgids, out_dir, args.subtask_num, g1_partitioner, count_discover, ctx_builder)
clients = []
for part_idx in range(1, args.subtask_num+1):
    conf = Config()
    part_data_dir = os.path.join(data_dir, f"partition_{part_idx}")
    part_out_dir = os.path.join(out_dir, f"partition_{part_idx}")
    conf.data_dir = part_data_dir
    conf.output_dir = part_out_dir
    conf.kgids = kgids
    conf.max_train_epoch = args.max_train_epoch
    conf.gcn_layer_num = args.layer_num
    conf.py_exe_fn = args.py_exe_fn
    conf.gpu_ids = args.gpu_ids
    conf.tf_gpu_id = int(args.gpu_ids[-1])
    conf.torch_device = torch.device("cuda:0")

    if args.ea_model == "rrea":
        ea_module = RREAModule(conf)
    else:
        ea_module = GCNAlignModule(conf)

    client = Client(part_data_dir, kgids, part_out_dir, ea_module)
    clients.append(client)


framework = ParallelEAFramework(server, clients, max_iteration=args.max_iteration)
framework.run()



