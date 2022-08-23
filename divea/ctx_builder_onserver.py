# -*- coding: utf-8 -*-

import os
import json
import torch
import sys
import nvidia_smi
nvidia_smi.nvmlInit()

from divea.components_base import ContextBuilder
from ctx_builder_onclient import compute_perf_jointly_multi_proc, compute_perf_multi_proc
from divea.misc import sub_alignment_with_head

# multiprocessing.set_start_method('spawn', force=True)
sys.path.append(os.getcwd())



class CtxBuilderV2(ContextBuilder):
    candi2benefit_map_list = None

    def __init__(self, data, data_dir, kgids, out_dir, part_n, gcn_l, gamma, ctx_g1_size=None, ctx_g2_size=None, ctx_g2_conn_percent=0.0, torch_devices=["cpu"]):
        super().__init__(data, data_dir, kgids, part_n, g1_ctx_only_once=True)
        self.gcn_l = gcn_l
        self.devices = torch_devices
        self.out_dir = out_dir

        self.gamma = gamma

        self.g1_ctx_size = ctx_g1_size
        self.g2_ctx_size = ctx_g2_size
        self.g2_conn_ent_percent = ctx_g2_conn_percent
        self.g2_conn_ent_num = int(self.g2_ctx_size * self.g2_conn_ent_percent)

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


        with open(os.path.join(self.data_dir, f"partition_{part_idx}/kg_partition.json")) as file:
            part_obj = json.loads(file.read())
            g1_part_entities = part_obj["kg1_partition"]["entities"]

        if part_idx == 1:
            labelled_alignmennt = self.data.load_train_alignment()
            pseudo_alignment = self.load_pseudo_seeds()
            self.train_alignment = labelled_alignmennt + pseudo_alignment

        train_alignment = self.train_alignment
        g1_anchors = [e1 for e1, e2 in train_alignment]
        ctx_entities = self._build_context_for_single_graph1(g1_part_entities, self.data.kg1_entities, self.g1_ctx_size, part_idx-1)

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

    def _build_context_for_single_graph1(self, unmatch_entities, entities, ctx_graph_size, part_idx):
        all_candidates = list(set(entities).difference(set(unmatch_entities)))
        max_drop_num = max(0, len(entities) - ctx_graph_size)
        candi2benefit_map = CtxBuilderV2.candi2benefit_map_list[part_idx]
        candi2benefit_map = {e: candi2benefit_map[e] for e in all_candidates}

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





