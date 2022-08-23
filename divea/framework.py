# -*- coding: utf-8 -*-


from divea.components_base import Server
from divea.components_base import Client
import typing
from divea.util import RunningLogger
import time
import json
import os
import numpy as np
import shutil
import torch.multiprocessing as mp

class ParallelEAFramework:
    def __init__(self, server: Server, clients: typing.List[Client], max_iteration=1):
        self.server = server
        self.clients = clients
        self.logger = RunningLogger(self.server.out_dir)
        self.data_dir = self.server.data_dir
        self.kgids = self.server.kgids
        self.out_dir = self.server.out_dir
        self.subtask_num = self.server.part_n
        self.max_iteration = max_iteration

    def _prepare(self):
        #
        for idx in range(1, self.subtask_num + 1):
            part_data_dir = os.path.join(self.data_dir, f"partition_{idx}")
            part_out_dir = os.path.join(self.out_dir, f"partition_{idx}")
            if os.path.exists(part_data_dir):
                shutil.rmtree(part_data_dir)
            if os.path.exists(part_out_dir):
                shutil.rmtree(part_out_dir)
            os.mkdir(part_data_dir)
            os.mkdir(part_out_dir)
        if os.path.exists(os.path.join(self.out_dir, "running.log")):
            os.remove(os.path.join(self.out_dir, "running.log"))
        if os.path.exists(os.path.join(self.out_dir, "tmp_running.log")):
            os.remove(os.path.join(self.out_dir, "tmp_running.log"))
        if os.path.exists(os.path.join(self.data_dir, "all_pseudo_seeds.txt")):
            os.remove(os.path.join(self.data_dir, "all_pseudo_seeds.txt"))

    def run(self):
        self._prepare()

        # partition G1
        print("begin partitioning")
        t11 = time.time()
        self.server.g1_partitioner.partition_g1_entities()
        t12 = time.time()
        msg_obj = {"machine": "server", "msg_type": "running_time", "value": t12 - t11, "process": "partition_g1"}
        print(msg_obj)
        self.logger.log(json.dumps(msg_obj))

        for ite in range(0, self.max_iteration):
            t60 = time.time()
            self.server.ctx_builder.build_g1_context()
            t61 = time.time()
            msg_obj = {"machine": "server", "msg_type": "running_time", "value": t61 - t60, "process": "build_context1", "iteration": ite}
            self.logger.log(json.dumps(msg_obj))

            # partition G2
            t21 = time.time()
            self.server.g2_partitioner.partition_g2_entities()
            t22 = time.time()
            msg_obj = {"machine": "server", "msg_type": "running_time", "value": t22 - t21, "process": "partition_g2",
                       "iteration": ite}
            self.logger.log(json.dumps(msg_obj))

            # build context
            t41 = time.time()
            self.server.ctx_builder.build_g2_context()
            t42 = time.time()
            msg_obj = {"machine": "server", "msg_type": "running_time", "value": t42 - t41, "process": "build_context2",
                       "iteration": ite}
            self.logger.log(json.dumps(msg_obj))

            # build sub graphs
            t51 = time.time()
            self.server.g1_partitioner.build_g1_subgraphs()
            self.server.g2_partitioner.build_g2_subgraphs()
            t52 = time.time()
            msg_obj = {"machine": "server", "msg_type": "running_time", "value": t52 - t51,
                       "process": "build_subgraphs", "iteration": ite}
            self.logger.log(json.dumps(msg_obj))

            for idx, client in enumerate(self.clients):
                # train model
                t31 = time.time()
                client.generate_dataset_from_partition()
                client.train_model()
                t32 = time.time()
                msg_obj = {"machine": f"client_{idx + 1}", "msg_type": "running_time", "value": t32 - t31,
                           "process": "train", "iteration": ite}
                self.logger.log(json.dumps(msg_obj))

                # evaluate model
                # client.evaluate_model()

                proc = mp.Process(target=client.evaluate_model)
                proc.start()
                proc.join()


                t33 = time.time()
                msg_obj = {"machine": f"client_{idx + 1}", "msg_type": "running_time", "value": t33 - t32,
                           "process": "evaluate", "iteration": ite}
                self.logger.log(json.dumps(msg_obj))

                # generate msg
                client.generate_msg()
                t34 = time.time()
                msg_obj = {"machine": f"client_{idx + 1}", "msg_type": "running_time", "value": t34 - t33,
                           "process": "generate_msg", "iteration": ite}
                self.logger.log(json.dumps(msg_obj))

            # communicate
            t51 = time.time()
            self.server.communicate()  # collect msgs from clients and deliver msgs to clients
            t52 = time.time()
            msg_obj = {"machine": f"server", "msg_type": "running_time", "value": t51 - t52,
                       "process": "communicate", "iteration": ite}
            self.logger.log(json.dumps(msg_obj))

        # self.evaluate()


        proc = mp.Process(target=self.evaluate)
        proc.start()
        proc.join()


    def evaluate(self):
        # with open(os.path.join(self.data_dir, "meta_info_part1.json")) as file:
        #     obj = json.loads(file.read())
        #     test_nums = obj["kg1_partition"]["test_align_nums"]

        test_nums = []
        for idx in range(len(self.clients)):
            part_data_dir = os.path.join(self.server.data_dir, f"partition_{idx + 1}")
            with open(os.path.join(part_data_dir, "kg_partition.json")) as file:
                obj = json.loads(file.read())
                test_n = len(obj["kg1_partition"]["test_alignment"])
                test_nums.append(test_n)

        recall_list = []
        mrr_list = []
        recall1_list = []
        recall5_list = []
        recall10_list = []
        recall50_list = []
        for idx in range(len(self.clients)):
            part_out_dir = os.path.join(self.server.out_dir, f"partition_{idx+1}")
            with open(os.path.join(part_out_dir, "part_metrics.json")) as file:
                cont = file.read().strip()
                last_line = cont.split("\n")[-1]
                obj = json.loads(last_line)
                recall = obj["recall"]
                recall_list.append(recall)
                mrr_list.append(obj["metrics_csls"]["mrr"] * recall)
                recall1_list.append(obj["metrics_csls"]["recall@1"] * recall)
                recall5_list.append(obj["metrics_csls"]["recall@5"] * recall)
                recall10_list.append(obj["metrics_csls"]["recall@10"] * recall)
                recall50_list.append(obj["metrics_csls"]["recall@50"] * recall)
        test_nums = np.array(test_nums)
        test_percents = test_nums / np.sum(test_nums)
        recall = np.matmul(test_percents, recall_list)
        mrr = np.matmul(test_percents, mrr_list)
        recall1 = np.matmul(test_percents, recall1_list)
        recall5 = np.matmul(test_percents, recall5_list)
        recall10 = np.matmul(test_percents, recall10_list)
        recall50 = np.matmul(test_percents, recall50_list)
        effective_metrics_obj = {
            "recall": recall,
            "mrr": mrr,
            "recall@1": recall1,
            "recall@5": recall5,
            "recall@10": recall10,
            "recall@50": recall50
        }

        running_obj = {
            "partition_g1": 0,
            "iterations": {
                0: {
                    "client_training": []
                }
            }
        }
        with open(os.path.join(self.server.out_dir, "running.log")) as file:
            cont = file.read().strip()
            lines = cont.split("\n")
            for line in lines:
                obj = json.loads(line)
                if obj["machine"] == "server" and obj["process"] == "partition_g1":
                    running_obj["partition_g1"] = obj["value"]
                elif obj["machine"].startswith("client"):
                    if obj["iteration"] not in running_obj["iterations"].keys():
                        running_obj["iterations"][obj["iteration"]] = {}
                    k = f'client_{obj["process"]}'
                    if k not in running_obj["iterations"][obj["iteration"]]:
                        running_obj["iterations"][obj["iteration"]][k] = []
                    running_obj["iterations"][obj["iteration"]][k].append(obj["value"])
                elif obj["machine"] == "server":
                    if obj["iteration"] not in running_obj["iterations"].keys():
                        running_obj["iterations"][obj["iteration"]] = {}
                    running_obj["iterations"][obj["iteration"]][obj["process"]] = obj["value"]

        running_metric_obj = self._collect_running_metrics(running_obj)


        # memory usage
        max_mem_list = []
        for idx in range(len(self.clients)):
            part_out_dir = os.path.join(self.server.out_dir, f"partition_{idx+1}")
            with open(os.path.join(part_out_dir, "running.log")) as file:
                lines = file.read().strip().split("\n")
                mem_before_list = []
                mem_after_list = []
                for line in lines:
                    obj = json.loads(line)
                    if obj["msg_type"] == "gpu_mem_usage_after":
                        mem_after_list.append(obj["value"])
                    if obj["msg_type"] == "gpu_mem_usage_before":
                        mem_before_list.append(obj["value"])
            mem = float(np.mean(mem_after_list) - np.mean(mem_before_list))
            max_mem_list.append(mem)
        max_mem = max(max_mem_list)

        if os.path.exists(os.path.join(self.out_dir, "tmp_running.log")):
            with open(os.path.join(self.out_dir, "tmp_running.log")) as file:
                lines = file.read().strip().split("\n")
                mem_before_list = []
                mem_after_list = []
                for line in lines:
                    obj = json.loads(line)
                    if obj["msg_type"] == "ctx1_gpu_mem_usage_after":
                        mem_after_list.append(obj["value"])
                    if obj["msg_type"] == "ctx1_gpu_mem_usage_before":
                        mem_before_list.append(obj["value"])
            ctx1_mem = float(np.mean(mem_after_list) - np.mean(mem_before_list))

            with open(os.path.join(self.out_dir, "tmp_running.log")) as file:
                lines = file.read().strip().split("\n")
            mem_before_list = []
            mem_after_list = []
            for line in lines:
                obj = json.loads(line)
                if obj["msg_type"] == "ctx2_gpu_mem_usage_after":
                    mem_after_list.append(obj["value"])
                if obj["msg_type"] == "ctx2_gpu_mem_usage_before":
                    mem_before_list.append(obj["value"])
            if len(mem_before_list) == 0:
                ctx2_mem = 0
            else:
                ctx2_mem = float(np.mean(mem_after_list) - np.mean(mem_before_list))
        else:
            ctx1_mem = -1
            ctx2_mem = -1

        metrics = {
            "effectiveness": effective_metrics_obj,
            "running_time": running_metric_obj,
            "ea_gpu_memory": max_mem,
            "ctx1_gpu_memory": ctx1_mem,
            "ctx2_gpu_memory": ctx2_mem
        }
        with open(os.path.join(self.server.out_dir, "metrics.json"), "w+") as file:
            file.write(json.dumps(metrics))

    @staticmethod
    def _collect_running_metrics(running_obj):
        part_g2_list = []
        build_context1 = []
        build_context2 = []
        build_subgraph =[]
        train_list = []
        for _, ite in running_obj["iterations"].items():
            part_g2_list.append(ite["partition_g2"])
            build_context1.append(ite["build_context1"])
            build_context2.append(ite["build_context2"])
            build_subgraph.append(ite["build_subgraphs"])
            train_list.append(max(ite["client_train"]))
        running_metric_obj = {
            "partition_g1": running_obj["partition_g1"],
            "partition_g2": part_g2_list,
            "build_context1": build_context1,
            "build_context2": build_context2,
            "build_subgraph": build_subgraph,
            "client_training": train_list,
            "total": running_obj["partition_g1"] + sum(part_g2_list) + sum(build_context1) + sum(build_context2) + sum(build_subgraph) + sum(train_list)        }
        return running_metric_obj








