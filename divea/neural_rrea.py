# -*- coding: utf-8 -*-

from divea.components_base import NeuralEAModule
from divea.util import Config
import os
import numpy as np
import json
from divea.data2 import read_alignment
from RREA.runner import Runner
import subprocess
from RREA.CSLS_torch import Evaluator
from divea.data2 import convert_uniform_to_rrea


class RREAModule(NeuralEAModule):
    def __init__(self, conf: Config):
        super(RREAModule, self).__init__(conf)
        # self.runner = Runner(self.conf.data_dir, self.conf.output_dir,
        #                 max_train_epoch=self.conf.max_train_epoch,
        #                 max_continue_epoch=self.conf.max_continue_epoch,
        #                 eval_freq=self.conf.eval_freq, depth=self.conf.gcn_layer_num)

    def refresh_weights(self):
        # self.runner.restore_model()
        pass

    def prepare_data(self):
        convert_uniform_to_rrea(self.conf.data_dir, self.conf.kgids)

    def train_model(self):
        if self.conf.py_exe_fn is None:
            runner = Runner(self.conf.data_dir, self.conf.output_dir,
                            max_train_epoch=self.conf.max_train_epoch,
                            depth=self.conf.gcn_layer_num,
                            tf_gpu_no=self.conf.gpu_ids)

            # runner.restore_model()
            runner.train()
            runner.save()
        else:
            cmd_fn = self.conf.py_exe_fn
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            script_fn = os.path.join(cur_dir, "../RREA/runner.py")
            args_str = f"--data_dir={self.conf.data_dir} --output_dir={self.conf.output_dir} " \
                       f"--max_train_epoch={self.conf.max_train_epoch} --layer_num={self.conf.gcn_layer_num} " \
                       f"--tf_gpu_no={self.conf.tf_gpu_id}"
            env = os.environ.copy()
            print(args_str)
            env["CUDA_VISIBLE_DEVICES"] = f"{self.conf.tf_gpu_id}"
            # env["CUDA_VISIBLE_DEVICES"] = "0,1"
            ret = subprocess.run(cmd_fn + " " + script_fn + " " + args_str, shell=True, env=env)
            if ret.returncode != 0:
                raise Exception("RREA did not run successfully.")

    def predict_simi(self):
        emb_res = np.load(os.path.join(self.conf.output_dir, "emb.npz"))
        embs = emb_res["embs"]
        ent1_ids = emb_res["ent1_ids"]
        ent2_ids = emb_res["ent2_ids"]
        ent1_embs = embs[ent1_ids]
        ent2_embs = embs[ent2_ids]

        evaluator = Evaluator(device=self.conf.torch_device)
        simi_mtx = evaluator.csls_sim(ent1_embs, ent2_embs, k=10)

        return simi_mtx

    def get_embeddings(self):
        emb_res = np.load(os.path.join(self.conf.output_dir, "emb.npz"))
        embs = emb_res["embs"]
        ent1_ids = emb_res["ent1_ids"]
        ent2_ids = emb_res["ent2_ids"]
        ent1_embs = embs[ent1_ids]
        ent2_embs = embs[ent2_ids]
        return ent1_embs, ent2_embs

    def get_pred_alignment(self):
        # emea_data = EMEAData(self.conf.data_dir, self.conf.data_name)
        with open(os.path.join(self.conf.output_dir, "pred_alignment.json")) as file:
            obj = json.loads(file.read())
            pred_alignment = obj["pred_alignment_csls"]
        # kg1_old2new_ent_map, kg2_old2new_ent_map = emea_data.old2new_entity_id_map()
        # pred_alignment = [(kg1_old2new_ent_map[ent1], kg2_old2new_ent_map[ent2]) for ent1, ent2 in pred_alignment]
        return pred_alignment


    def evaluate(self):
        emb_res = np.load(os.path.join(self.conf.output_dir, "emb.npz"))
        embs = emb_res["embs"]
        ent1_ids = emb_res["ent1_ids"]
        ent2_ids = emb_res["ent2_ids"]
        ent1_embs = embs[ent1_ids]
        ent2_embs = embs[ent2_ids]

        eval_alignment = read_alignment(os.path.join(self.conf.data_dir, "test_alignment.txt"))

        evaluator = Evaluator(device=self.conf.torch_device)
        cos_test_metrics, cos_test_alignment = evaluator.evaluate_cosine(ent1_embs, ent2_embs, eval_alignment)
        csls_test_metrics, csls_test_alignment = evaluator.evaluate_csls(ent1_embs, ent2_embs, eval_alignment)
        csls_all_alignment = evaluator.predict_alignment(ent1_embs, ent2_embs)

        metrics_obj = {"metrics_csls": csls_test_metrics, "metrics_cos": cos_test_metrics}
        print("csls metrics: ", csls_test_metrics)
        pred_alignment_obj = {"pred_alignment_csls": csls_test_alignment.tolist(),
                              "pred_alignment_cos": cos_test_alignment.tolist(),
                              "all_pred_alignment_csls": csls_all_alignment.tolist()
                              }
        with open(os.path.join(self.conf.output_dir, "metrics.json"), "w+") as file:
            file.write(json.dumps(metrics_obj))
        with open(os.path.join(self.conf.output_dir, "pred_alignment.json"), "w+") as file:
            file.write(json.dumps(pred_alignment_obj))
        with open(os.path.join(self.conf.output_dir, "eval_metrics.json"), "w+") as file:
            file.write(json.dumps(csls_test_metrics))
        return metrics_obj

    def evaluate_given_alignment(self, eval_alignment):
        emb_res = np.load(os.path.join(self.conf.output_dir, "emb.npz"))
        embs = emb_res["embs"]
        ent1_ids = emb_res["ent1_ids"]
        ent2_ids = emb_res["ent2_ids"]
        ent1_embs = embs[ent1_ids]
        ent2_embs = embs[ent2_ids]

        evaluator = Evaluator(device=self.conf.torch_device)
        cos_test_metrics, cos_test_alignment = evaluator.evaluate_cosine(ent1_embs, ent2_embs, eval_alignment)
        csls_test_metrics, csls_test_alignment = evaluator.evaluate_csls(ent1_embs, ent2_embs, eval_alignment)

        metrics_obj = {"metrics_csls": csls_test_metrics, "metrics_cos": cos_test_metrics}
        print("csls metrics: ", csls_test_metrics)
        return csls_test_metrics, cos_test_metrics



