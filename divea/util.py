# -*- coding: utf-8 -*-


import random
import os
import numpy as np
import tensorflow as tf
import torch


def seed_everything(seed=1011):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class RunningLogger:
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def log(self, msg: str):
        with open(os.path.join(self.out_dir, "running.log"), "a+") as file:
            file.write(msg.strip() + "\n")



class Config:
    def __init__(self, fn=None):
        # files and directory
        self.data_dir = None
        self.output_dir = None
        self.kgids = None

        # running env
        self.gpu_ids = None
        self.tf_gpu_id = None
        self.torch_device = None

        self.gcn_layer_num = None
        self.py_exe_fn = None
        self.max_train_epoch = None

