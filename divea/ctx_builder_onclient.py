# -*- coding: utf-8 -*-
import time


from tqdm import tqdm
import torch.nn as nn
import sys
import dgl
import numpy as np
from torch import multiprocessing
import dgl.function as fn

import os
import json
import torch
import nvidia_smi
nvidia_smi.nvmlInit()

from divea.components_base import ContextBuilder
# from ctx_builder_onclient import compute_perf_jointly_multi_proc, compute_perf_multi_proc



# multiprocessing.set_start_method('spawn')
multiprocessing.get_context("spawn")
sys.path.append(os.getcwd())


class EviPassingLayer(nn.Module):
    def __init__(self):
        super(EviPassingLayer, self).__init__()

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.update_all(message_func=fn.copy_u('h','m'), reduce_func=fn.sum("m", "o"))
            h_o = g.ndata["o"]
            return h_o


class EviPassingModel(nn.Module):
    def __init__(self, layer_num):
        super(EviPassingModel, self).__init__()
        self.layer_num = layer_num
        self.layer = EviPassingLayer()

    def forward(self, g, h0, e):
        h = h0
        for i in range(self.layer_num):
            h = h * e
            h = self.layer(g, h)
        h = h * e
        return h


class PerfModel():
    def __init__(self, ent_num, kg_entities, kg_triples, device, gcn_l, gamma=2.0):
        triple_arr = np.array(kg_triples)
        self.gamma = gamma
        self.gcn_l = gcn_l
        # triple_arr = torch.tensor(kg_triples)
        # t1 = triple_arr[:,0]
        # t2 = triple_arr[:, 2]
        # self.graph = dgl.graph()
        self.graph = dgl.DGLGraph()
        self.graph.add_edges(triple_arr[:,0], triple_arr[:,2])
        self.graph.add_edges(triple_arr[:,2], triple_arr[:, 0])
        self.graph.add_edges(np.array(kg_entities), np.array(kg_entities))
        self.ent_num = ent_num

    def compute_perf_jointly(self, candi_entities, anc_entities, unmatch_entities_list, device):
        device = torch.device(device)
        model = EviPassingModel(layer_num=self.gcn_l).to(device)
        # entity existence
        e = torch.ones(size=(self.ent_num,), device=device)
        # graph
        graph = self.graph.to(device)
        # compute he_to_inanchorpath_num
        h0 = torch.zeros(size=(self.ent_num,), device=device)
        h0[anc_entities] = 1
        init_ent_inanchorpath_nums = model(graph, h0, e)  # + 1e-8
        ent_inanchorpath_ratios = init_ent_inanchorpath_nums / (init_ent_inanchorpath_nums+1e-8)
        init_weight_perf = 2 / (1+torch.exp(-self.gamma*ent_inanchorpath_ratios)) - 1
        # compute he_to_inpath_num
        init_ent_inpath_nums = model(graph, init_weight_perf, e) # + 1e-8
        ent_inpath_ratios = init_ent_inpath_nums / (init_ent_inpath_nums+1e-8)
        emb_perf = 2 / (1+torch.exp(-self.gamma*ent_inpath_ratios)) - 1

        ori_perf_list = []
        for unmatch_entities in unmatch_entities_list:
            ori_perf = torch.sum(emb_perf[unmatch_entities])
            ori_perf_list.append(ori_perf)

        candi2benefit_map_list = [dict() for _ in range(len(unmatch_entities_list))]
        with torch.no_grad():
            h0 = torch.zeros(size=(self.ent_num,), device=device)
            h0[anc_entities] = 1
            for candi in tqdm(candi_entities):
                e[candi] = 0
                ent_inanchorpath_nums = model(graph, h0, e)
                ent_inanchorpath_ratios = ent_inanchorpath_nums / (init_ent_inanchorpath_nums+1e-8)
                weight_perf = 2 / (1+torch.exp(-self.gamma*ent_inanchorpath_ratios)) - 1
                ent_inpath_nums = model(graph, weight_perf, e)
                ent_inpath_ratios = ent_inpath_nums / (init_ent_inpath_nums+1e-8)
                emb_perf = 2 / (1+torch.exp(-self.gamma*ent_inpath_ratios)) - 1
                e[candi] = 1
                for idx, unmatch_entities in enumerate(unmatch_entities_list):
                    perf = torch.sum(emb_perf[unmatch_entities])
                    # candi2benefit_map_list[idx][candi] = (perf - ori_perf_list[idx]).cpu().item()
                    candi2benefit_map_list[idx][candi] = (ori_perf_list[idx] - perf).cpu().item()
        return candi2benefit_map_list

    def compute_perf(self, candi_entities, anc_entities, unmatch_entities, device):
        device = torch.device(device)
        model = EviPassingModel(layer_num=self.gcn_l).to(device)
        # entity existence
        e = torch.ones(size=(self.ent_num,), device=device)
        # graph
        graph = self.graph.to(device)
        # compute he_to_inanchorpath_num
        h0 = torch.zeros(size=(self.ent_num,), device=device)
        h0[anc_entities] = 1
        init_ent_inanchorpath_nums = model(graph, h0, e)  # + 1e-8
        ent_inanchorpath_ratios = init_ent_inanchorpath_nums / (init_ent_inanchorpath_nums+1e-8)
        init_weight_perf = 2 / (1+torch.exp(-self.gamma*ent_inanchorpath_ratios)) - 1
        # compute he_to_inpath_num
        init_ent_inpath_nums = model(graph, init_weight_perf, e) # + 1e-8

        ent_inpath_ratios = init_ent_inpath_nums / (init_ent_inpath_nums+1e-8)
        emb_perf = 2 / (1+torch.exp(-self.gamma*ent_inpath_ratios)) - 1
        tmp = emb_perf[unmatch_entities]
        ori_perf = torch.sum(emb_perf[unmatch_entities])

        candi2benefit_map = dict()
        with torch.no_grad():
            h0 = torch.zeros(size=(self.ent_num,), device=device)
            h0[anc_entities] = 1
            for candi in tqdm(candi_entities):
                e[candi] = 0
                ent_inanchorpath_nums = model(graph, h0, e)
                ent_inanchorpath_ratios = ent_inanchorpath_nums / (init_ent_inanchorpath_nums+1e-8)
                weight_perf = 2 / (1+torch.exp(-self.gamma*ent_inanchorpath_ratios)) - 1
                ent_inpath_nums = model(graph, weight_perf, e)
                ent_inpath_ratios = ent_inpath_nums / (init_ent_inpath_nums+1e-8)
                emb_perf = 2 / (1+torch.exp(-self.gamma*ent_inpath_ratios)) - 1
                e[candi] = 1
                perf = torch.sum(emb_perf[unmatch_entities])
                # candi2benefit_map[candi] = (perf - ori_perf).cpu().item()  # perf change of dropping entity
                candi2benefit_map[candi] = (ori_perf - perf).cpu().item()  # effect
        return candi2benefit_map


def compute_perf_jointly_multi_proc(ent_num, kg_entities, kg_triples, gcn_l, gamma,  candi_entities, anc_entities, unmatch_entities_list, device_list):
    global single_proc
    def single_proc(i):
        device = device_list[i]
        if device == "cpu":
            torch.set_num_threads(20)
        model = PerfModel(ent_num, kg_entities, kg_triples, device=device, gcn_l=gcn_l, gamma=gamma)
        st = batch_size * i
        ed = batch_size * (i+1)
        sub_candi_entities = candi_entities[st:ed]
        candi2benefit_map_list = model.compute_perf_jointly(sub_candi_entities, anc_entities, unmatch_entities_list, device)
        return candi2benefit_map_list

    proc_n = len(device_list)
    print("process num:", len(device_list))
    print(device_list)
    batch_size = int(len(candi_entities) / proc_n) + 1
    with multiprocessing.Pool(processes=proc_n) as pool:
        results = pool.map(single_proc, list(range(proc_n)))
    # pool.close()

    all_candi2benefit_map_list = [dict() for _ in range(len(unmatch_entities_list))]
    for res_list in results:
        for idx, res in enumerate(res_list):
            all_candi2benefit_map_list[idx].update(res)
    return all_candi2benefit_map_list


def compute_perf_multi_proc1(ent_num, kg_entities, kg_triples, gcn_l, gamma,  candi_entities, anc_entities, unmatch_entities, device_list):
    global single_proc
    def single_proc(i):
        device = device_list[i]
        if device == "cpu":
            torch.set_num_threads(20)
        model = PerfModel(ent_num, kg_entities, kg_triples, device=device, gcn_l=gcn_l, gamma=gamma)
        st = batch_size * i
        ed = batch_size * (i+1)
        sub_candi_entities = candi_entities[st:ed]
        candi2benefit_map = model.compute_perf(sub_candi_entities, anc_entities, unmatch_entities, device)
        return candi2benefit_map

    proc_n = len(device_list)
    print("process num:", len(device_list))
    print(device_list)
    batch_size = int(len(candi_entities) / proc_n) + 1
    # pool = multiprocessing.Pool(processes=proc_n)
    # results = pool.map(single_proc, list(range(proc_n)))
    # pool.close()
    # all_candi2benefit_map = dict()
    # for res in results:
    #     all_candi2benefit_map.update(res)

    all_candi2benefit_map = single_proc(0)
    return all_candi2benefit_map



def compute_perf_multi_proc(ent_num, kg_entities, kg_triples, gcn_l, gamma,  candi_entities, anc_entities, unmatch_entities, device_list):
    global single_proc
    def single_proc(i):
        device = device_list[i]
        if device == "cpu":
            torch.set_num_threads(20)
        model = PerfModel(ent_num, kg_entities, kg_triples, device=device, gcn_l=gcn_l, gamma=gamma)
        st = batch_size * i
        ed = batch_size * (i+1)
        sub_candi_entities = candi_entities[st:ed]
        candi2benefit_map = model.compute_perf(sub_candi_entities, anc_entities, unmatch_entities, device)
        return candi2benefit_map

    proc_n = len(device_list)
    print("process num:", len(device_list))
    print(device_list)
    batch_size = int(len(candi_entities) / proc_n) + 1
    with multiprocessing.Pool(processes=proc_n) as pool:
        results = pool.map(single_proc, list(range(proc_n)))

    all_candi2benefit_map = dict()
    for res in results:
        all_candi2benefit_map.update(res)

    return all_candi2benefit_map



class CtxBuilderV1(ContextBuilder):
    def __init__(self, data, data_dir, kgids, out_dir, subtask_num, gcn_l, ctx_g1_size, ctx_g2_size, gamma=2, ctx_g2_conn_percent=0.0, torch_devices=["cpu"]):
        super().__init__(data, data_dir, kgids, subtask_num, 0.0, g1_ctx_only_once=True)
        self.gcn_l = gcn_l
        self.devices = torch_devices
        self.out_dir = out_dir
        self.gamma = gamma
        self.g1_ctx_size = ctx_g1_size
        self.g2_ctx_size = ctx_g2_size
        self.g2_conn_ent_percent = ctx_g2_conn_percent
        self.g2_conn_ent_num = int(self.g2_ctx_size * self.g2_conn_ent_percent)

    def _build_g1_context(self, part_idx):
        with open(os.path.join(self.data_dir, f"partition_{part_idx}/kg_partition.json")) as file:
            part_obj = json.loads(file.read())
            g1_part_entities = part_obj["kg1_partition"]["entities"]

        if part_idx == 1:
            labelled_alignmennt = self.data.load_train_alignment()
            pseudo_alignment = self.load_pseudo_seeds()
            self.train_alignment = labelled_alignmennt + pseudo_alignment

        train_alignment = self.train_alignment
        g1_anchors = [e1 for e1, e2 in train_alignment]
        ctx_entities = self._build_context_for_single_graph1(g1_part_entities, self.data.kg1_entities, self.data.kg1_triples, g1_anchors, self.g1_ctx_size, part_idx)

        #
        part_data_dir = os.path.join(self.data_dir, f"partition_{part_idx}")
        with open(os.path.join(part_data_dir, "kg_partition.json")) as file:
            obj = json.loads(file.read())
        obj["kg1_partition"]["ctx_entities"] = ctx_entities
        ctx_anchors = list(set(g1_anchors).intersection(set(ctx_entities)))
        train_align_map = dict(train_alignment)
        ctx_train_alignment = [(anc, train_align_map[anc]) for anc in ctx_anchors]
        obj["kg1_partition"]["ctx_train_alignment"] = ctx_train_alignment
        with open(os.path.join(part_data_dir, "kg_partition.json"), "w+") as file:
            file.write(json.dumps(obj))

    def _build_g2_context(self, part_idx):
        with open(os.path.join(self.data_dir, f"partition_{part_idx}/kg_partition.json")) as file:
            part_obj = json.loads(file.read())
            g2_part_entities = part_obj["kg2_partition"]["entities"]

        if part_idx == 1:
            labelled_alignmennt = self.data.load_train_alignment()
            pseudo_alignment = self.load_pseudo_seeds()
            self.train_alignment = labelled_alignmennt + pseudo_alignment

        train_alignment = self.train_alignment
        g1_entities = part_obj["kg1_partition"]["entities"] + part_obj["kg1_partition"]["ctx_entities"]
        g1_ent2exist_map = {e:True for e in g1_entities}
        train_alignment = [(e1,e2) for e1, e2 in train_alignment if e1 in g1_ent2exist_map]

        g2_anchors = list(set([e2 for e1, e2 in train_alignment]))
        g2_anchor_map = {e: True for e in g2_anchors}
        g2_part_entities = [e for e in g2_part_entities if e not in g2_anchor_map]
        g2_part_entities = g2_part_entities[:(self.g2_ctx_size - self.g2_conn_ent_num - len(g2_anchors))]
        link_entities = self._build_context_for_single_graph2(g2_part_entities, self.data.kg2_entities, self.data.kg2_triples, g2_anchors, self.g2_conn_ent_num)
        ctx_entities = g2_anchors + link_entities

        # self._save_context(part_idx, g1_ctx_entities=None, g2_ctx_entities=ctx_entities)
        part_data_dir = os.path.join(self.data_dir, f"partition_{part_idx}")
        with open(os.path.join(part_data_dir, "kg_partition.json")) as file:
            obj = json.loads(file.read())
            obj["kg2_partition"]["ctx_entities"] = ctx_entities
            obj["kg2_partition"]["entities"] = g2_part_entities
        with open(os.path.join(part_data_dir, "kg_partition.json"), "w+") as file:
            file.write(json.dumps(obj))


    def _save_context(self, part_idx, g1_ctx_entities, g2_ctx_entities):
        part_data_dir = os.path.join(self.data_dir, f"partition_{part_idx}")
        with open(os.path.join(part_data_dir, "kg_partition.json")) as file:
            obj = json.loads(file.read())
        if g1_ctx_entities is not None:
            obj["kg1_partition"]["ctx_entities"] = g1_ctx_entities
        if g2_ctx_entities is not None:
            obj["kg2_partition"]["ctx_entities"] = g2_ctx_entities
        with open(os.path.join(part_data_dir, "kg_partition.json"), "w+") as file:
            file.write(json.dumps(obj))

    def _filter_g1(self, entities, triples, anchors, unmatch_entities, max_hop_k, part_idx):
        with open(os.path.join(self.data_dir, f"partition_{part_idx}/tmp_part.json")) as file:
            graph_partition_entities = json.loads(file.read())
        nei_ent_set = self.get_neighbours4(triples, graph_partition_entities, anchors, max_hop_k=max_hop_k)
        inv_ent_set = set(graph_partition_entities)
        inv_ent_set.update(nei_ent_set)
        candidate_set = inv_ent_set.difference(set(unmatch_entities))
        fil_entities = list(inv_ent_set)
        # fil_triples = subgraph(fil_entities, triples)
        fil_triples = self.data.kg1_sub_triples(fil_entities)
        fil_anchors = list(set(anchors).intersection(set(fil_entities)))
        return fil_entities, fil_triples, fil_anchors, candidate_set

    def _build_context_for_single_graph1(self, unmatch_entities, entities, triples, anchors, ctx_graph_size, part_idx):
        fil_entities, fil_triples, fil_anchors, all_candidates = self._filter_g1(entities, triples, anchors, unmatch_entities, max_hop_k=1, part_idx=part_idx)

        old2new_entid_map = {e: idx for idx, e in enumerate(fil_entities)}
        new2old_entid_map = {v:k for k,v in old2new_entid_map.items()}
        new_entities = [old2new_entid_map[e] for e in fil_entities]
        new_triples = [(old2new_entid_map[h], r, old2new_entid_map[t]) for h,r,t in fil_triples]
        new_anchors = [old2new_entid_map[e] for e in fil_anchors]
        new_candidates = [old2new_entid_map[e] for e in all_candidates]
        new_unmatch_entities = [old2new_entid_map[e] for e in unmatch_entities]
        new_candi2benefit_map = compute_perf_multi_proc(len(new_entities), new_entities, new_triples, self.gcn_l, self.gamma, new_candidates, new_anchors, new_unmatch_entities, self.devices)
        candi2benefit_map = {new2old_entid_map[k]:v for k,v in new_candi2benefit_map.items()}

        # max_drop_num = max(0, len(all_candidates) - (ctx_graph_size-len(unmatch_entities)))
        # sorted_items = sorted(candi2benefit_map.items(), key=lambda item: - item[1])
        # dropped_entities = [c for c,v in sorted_items[:max_drop_num]]  # drop front ones
        # sel_entities = list(set(all_candidates).difference(set(dropped_entities)))

        sel_num = max(0, ctx_graph_size-len(unmatch_entities))
        new_candi2effect_map = candi2benefit_map
        sorted_items = sorted(new_candi2effect_map.items(), key=lambda item: - item[1])
        sel_entities = [c for c,v in sorted_items[:sel_num]]  # drop front ones

        # del perfmodel
        torch.cuda.empty_cache()
        print("end of building g1 ctx")
        return sel_entities

    def _filter_g2(self, entities, triples, anchors, unmatch_entities, max_hop_k):
        nei_list = self.get_neighbours2(triples, anchors, max_hop_k=max_hop_k)
        nei_ent_set = set()
        for neis in nei_list:
            nei_ent_set.update(neis)
        inv_ent_set = set(unmatch_entities+anchors)
        inv_ent_set.update(nei_ent_set)
        candidate_set = nei_ent_set.difference(set(unmatch_entities+anchors))
        fil_entities = list(inv_ent_set)
        # fil_triples = subgraph(fil_entities, triples)
        fil_triples = self.data.kg2_sub_triples(fil_entities)
        fil_anchors = list(set(anchors).intersection(set(fil_entities)))
        return fil_entities, fil_triples, fil_anchors, candidate_set


    def _build_context_for_single_graph2(self, unmatch_entities, entities, triples, anchors, sel_num):
        if sel_num == 0:
            print("skip building ctx G2")
            return []

        fil_entities, fil_triples, fil_anchors, all_candidates = self._filter_g2(entities, triples, anchors, unmatch_entities, max_hop_k=1)

        old2new_entid_map = {e: idx for idx, e in enumerate(fil_entities)}
        new2old_entid_map = {v:k for k,v in old2new_entid_map.items()}
        new_entities = [old2new_entid_map[e] for e in fil_entities]
        new_triples = [(old2new_entid_map[h], r, old2new_entid_map[t]) for h,r,t in fil_triples]
        new_anchors = [old2new_entid_map[e] for e in fil_anchors]
        new_candidates = [old2new_entid_map[e] for e in all_candidates]
        new_unmatch_entities = [old2new_entid_map[e] for e in unmatch_entities]
        new_candi2effect_map = compute_perf_multi_proc(len(new_entities), new_entities, new_triples, self.gcn_l, self.gamma, new_candidates, new_anchors, new_unmatch_entities, self.devices)
        candi2effect_map = {new2old_entid_map[k]:v for k,v in new_candi2effect_map.items()}

        sorted_items = sorted(candi2effect_map.items(), key=lambda item: - item[1])
        sel_entities = [c for c,v in sorted_items[:sel_num]]  # drop front ones

        # del perfmodel
        torch.cuda.empty_cache()
        print("end of building g2 ctx")
        return sel_entities



class CtxBuilderV2(CtxBuilderV1):
    candi2benefit_map_list = None

    def __init__(self, data, data_dir, kgids, out_dir, subtask_num, gcn_l, gamma, ctx_g1_size=None, ctx_g2_size=None, ctx_g2_conn_percent=0.0, torch_devices=["cpu"]):
        super().__init__(data, data_dir, kgids, out_dir, subtask_num, gcn_l, ctx_g1_size, ctx_g2_size, gamma=gamma, ctx_g2_conn_percent=ctx_g2_conn_percent, torch_devices=torch_devices)

    def cache_g1_ent_effect(self):
        print("cache g1 ent perf")
        gpu_no = int(self.devices[0][-1])
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_no)
        info_before = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        unmatched_entities_list = []
        for part_idx in range(1, self.part_n+1):
            with open(os.path.join(self.data_dir, f"partition_{part_idx}/kg_partition.json")) as file:
                part_obj = json.loads(file.read())
                g1_part_entities = part_obj["kg1_partition"]["entities"]
                unmatched_entities_list.append(g1_part_entities)

        labelled_alignmennt = self.data.load_train_alignment()
        pseudo_alignment = self.load_pseudo_seeds()
        train_alignment = labelled_alignmennt + pseudo_alignment
        g1_anchors = [e1 for e1, e2 in train_alignment]

        candi2benefit_map_list = compute_perf_jointly_multi_proc(len(self.data.kg1_entities), self.data.kg1_entities, self.data.kg1_triples, gcn_l=self.gcn_l, gamma=self.gamma,
                                                                 candi_entities=self.data.kg1_entities, anc_entities=g1_anchors, unmatch_entities_list=unmatched_entities_list, device_list=self.devices)

        info_after = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        with open(os.path.join(self.out_dir, "tmp_running.log"), "a+") as file:
            msg1 = {"msg_type": "ctx1_gpu_mem_usage_before", "value": info_before.used/1024/1024}
            file.write(json.dumps(msg1)+"\n")
            msg2 = {"msg_type": "ctx1_gpu_mem_usage_after", "value": info_after.used/1024/1024}
            file.write(json.dumps(msg2)+"\n")
        return candi2benefit_map_list

    def _build_g1_context(self, part_idx):
        if CtxBuilderV2.candi2benefit_map_list is None:
            CtxBuilderV2.candi2benefit_map_list = self.cache_g1_ent_effect()

        super()._build_g1_context(part_idx)

    def _build_context_for_single_graph1(self, unmatch_entities, entities, triples, anchors, ctx_graph_size, part_idx):
        all_candidates = list(set(entities).difference(set(unmatch_entities)))
        candi2benefit_map = CtxBuilderV2.candi2benefit_map_list[part_idx-1]
        candi2benefit_map = {e: candi2benefit_map[e] for e in all_candidates}

        sel_num = max(0, ctx_graph_size-len(unmatch_entities))
        new_candi2effect_map = candi2benefit_map
        sorted_items = sorted(new_candi2effect_map.items(), key=lambda item: - item[1])
        sel_entities = [c for c,v in sorted_items[:sel_num]]  # drop front ones

        # del perfmodel
        torch.cuda.empty_cache()
        print("end of building g1 ctx")
        return sel_entities






