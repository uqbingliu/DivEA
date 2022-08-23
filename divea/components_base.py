# -*- coding: utf-8 -*-

import os
import abc
import json
import shutil
import networkx as nx
import numpy as np
import torch
import pandas as pd
from tqdm import trange, tqdm

from divea.util import Config
from divea.data2 import read_tab_lines, write_tab_lines
from divea.data2 import UniformData
from divea.data2 import read_alignment
from RREA.CSLS_torch import Evaluator
from divea.misc import sub_alignment_with_head, sub_alignment_with_tail


class NeuralEAModule:
    def __init__(self, conf: Config):
        self.conf = conf

    @abc.abstractmethod
    def prepare_data(self):
        pass

    @abc.abstractmethod
    def train_model(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    @abc.abstractmethod
    def get_embeddings(self):
        pass

    @abc.abstractmethod
    def get_pred_alignment(self):
        pass

    def get_all_pred_alignment(self):
        with open(os.path.join(self.conf.output_dir, "pred_alignment.json")) as file:
            obj = json.loads(file.read())
            pred_alignment = obj["pred_alignment_csls"]
            all_pred_alignment = obj["all_pred_alignment_csls"]
        return pred_alignment, all_pred_alignment

    @abc.abstractmethod
    def evaluate(self):
        pass

    @abc.abstractmethod
    def evaluate_given_alignment(self, eval_alignment):
        pass


class Client:
    def __init__(self, part_data_dir, kgids, part_out_dir, ea_module: NeuralEAModule):
        self.subtask_data_dir = part_data_dir
        self.subtask_out_dir = part_out_dir
        self.ea_module = ea_module
        self.kgids = kgids

    def train_model(self):
        self.ea_module.prepare_data()
        self.ea_module.train_model()

    def generate_dataset_from_partition(self):
        data_dir = os.path.dirname(self.subtask_data_dir)

        kgid1, kgid2 = self.kgids

        with open(os.path.join(self.subtask_data_dir, "kg_partition.json")) as file:
            obj = json.loads(file.read())
            kg1_part_obj = obj["kg1_partition"]
            kg2_part_obj = obj["kg2_partition"]

        # whole_data = UniformData(data_dir, kgids)
        kg1_old2new_id_map = dict()
        kg1_entities = list()
        for idx, oldid in enumerate(sorted(list(set(kg1_part_obj["entities"]+kg1_part_obj["ctx_entities"])))):
            kg1_old2new_id_map[oldid] = idx
            kg1_entities.append((idx, oldid))
        kg1_triples = list()
        for h, r, t in kg1_part_obj["triples"]:  #+kg1_part_obj["ctx_triples"]:
            kg1_triples.append((kg1_old2new_id_map[h], r, kg1_old2new_id_map[t]))

        write_tab_lines(kg1_entities, os.path.join(self.subtask_data_dir, f"{kgid1}_entity_id2uri.txt"))
        write_tab_lines(kg1_triples, os.path.join(self.subtask_data_dir, f"{kgid1}_triple_rel.txt"))


        kg2_old2new_id_map = dict()
        kg2_entities = list()
        for idx, oldid in enumerate(kg2_part_obj["entities"]):
            kg2_old2new_id_map[oldid] = idx
            kg2_entities.append((idx, oldid))
        kg2_triples = list()
        for h, r, t in kg2_part_obj["triples"]:
            kg2_triples.append((kg2_old2new_id_map[h], r, kg2_old2new_id_map[t]))

        alignment = [(kg1_old2new_id_map[e1], kg2_old2new_id_map[e2]) for e1, e2 in kg2_part_obj["alignment"]]
        all_train_alignment = kg2_part_obj["train_alignment"]  #+kg2_part_obj["ctx_train_alignment"]
        if "valid_pseudo_seeds" in kg2_part_obj:
            all_train_alignment += kg2_part_obj["valid_pseudo_seeds"]
        else:
            pseudo_alignment = self.load_pseudo_seeds()
            all_train_alignment += pseudo_alignment
        train_alignment = [(kg1_old2new_id_map[e1], kg2_old2new_id_map[e2]) for e1, e2 in all_train_alignment]
        test_alignment = [(kg1_old2new_id_map[e1], kg2_old2new_id_map[e2]) for e1, e2 in kg2_part_obj["test_alignment"]]

        write_tab_lines(kg2_entities, os.path.join(self.subtask_data_dir, f"{kgid2}_entity_id2uri.txt"))
        write_tab_lines(kg2_triples, os.path.join(self.subtask_data_dir, f"{kgid2}_triple_rel.txt"))
        write_tab_lines(alignment, os.path.join(self.subtask_data_dir, "alignment_of_entity.txt"))
        write_tab_lines(train_alignment, os.path.join(self.subtask_data_dir, "train_alignment.txt"))
        write_tab_lines(test_alignment, os.path.join(self.subtask_data_dir, "test_alignment.txt"))

        shutil.copy(os.path.join(data_dir, f"{kgid2}_relation_id2uri.txt"),
                    os.path.join(self.subtask_data_dir, f"{kgid2}_relation_id2uri.txt"))
        shutil.copy(os.path.join(data_dir, f"{kgid1}_relation_id2uri.txt"),
                    os.path.join(self.subtask_data_dir, f"{kgid1}_relation_id2uri.txt"))

    def evaluate_model(self):
        metrics_obj = self.ea_module.evaluate()

        with open(os.path.join(self.subtask_data_dir, "kg_partition.json")) as file:
            obj = json.loads(file.read())
            all_test_alignment = obj["kg1_partition"]["test_alignment"]
            valid_test_alignment = obj["kg2_partition"]["test_alignment"]

        recall = len(valid_test_alignment) / len(all_test_alignment)

        # metrics_obj = {"recall": recall, "metrics_csls": metrics_obj["metrics_csls"], "metrics_cos": metrics_obj["metrics_cos"] }
        metrics_obj = {"recall": recall, "metrics_csls": metrics_obj["metrics_csls"] }
        with open(os.path.join(self.subtask_out_dir, "part_metrics.json"), "a+") as file:
            file.write(json.dumps(metrics_obj)+"\n")


        ## for analyzing partG2
        if os.path.exists(os.path.join(self.subtask_out_dir, "emb.npz")):
            features = self._get_features(self.subtask_data_dir, self.subtask_out_dir)
            tmp_fea_fn = os.path.join(self.subtask_data_dir, f"tmp_fea.txt")
            with open(tmp_fea_fn, "a+") as file:
                file.write(json.dumps(features) + "\n")

    def _get_features(self, part_data_dir, part_out_dir):
        self.device = "cuda:0"
        emb_fn = os.path.join(part_out_dir, "emb.npz")
        emb_res = np.load(emb_fn)
        embs = emb_res["embs"]
        ent1_ids = emb_res["ent1_ids"]
        ent2_ids = emb_res["ent2_ids"]
        ent1_embs = embs[ent1_ids]
        ent2_embs = embs[ent2_ids]

        graphid = "g2"
        if graphid == "g1":
            src_embs = ent1_embs
            dst_embs = ent2_embs
        else:
            src_embs = ent2_embs
            dst_embs = ent1_embs


        evaluator = Evaluator(device=self.device)
        sim_mtx = evaluator.csls_sim(src_embs, dst_embs)
        batch_size = 512

        with torch.no_grad():
            total_size = sim_mtx.shape[0]
            score1_list = []
            score5_list = []
            score10_list = []
            score15_list = []
            score20_list = []
            for cursor in range(0, total_size, batch_size):
                if isinstance(sim_mtx, np.ndarray):
                    sub_sim_mtx = sim_mtx[cursor: cursor + batch_size]
                    sub_sim_mtx = torch.tensor(sub_sim_mtx, device=self.device)
                else:
                    sub_sim_mtx = sim_mtx[cursor: cursor + batch_size].to(self.device)

                sorted_sim, _ = torch.sort(sub_sim_mtx, dim=1, descending=True)

                scores1 = sorted_sim[:, 0]
                scores5 = torch.sum(sorted_sim[:, 0:5], dim=1)
                scores10 = torch.sum(sorted_sim[:, 0:10], dim=1)
                scores15 = torch.sum(sorted_sim[:, 0:15], dim=1)
                scores20 = torch.sum(sorted_sim[:, 0:20], dim=1)

                score1_list.extend(scores1.cpu().tolist())
                score5_list.extend(scores5.cpu().tolist())
                score10_list.extend(scores10.cpu().tolist())
                score15_list.extend(scores15.cpu().tolist())
                score20_list.extend(scores20.cpu().tolist())

        if graphid == "g1":
            src_ent_ids = ent1_ids
            map1_fn = os.path.join(part_data_dir, "ent_ids_1")
            map2_fn = os.path.join(part_data_dir, f"{self.kgids[0]}_entity_id2uri.txt")
        else:
            src_ent_ids = ent2_ids
            map1_fn = os.path.join(part_data_dir, "ent_ids_2")
            map2_fn = os.path.join(part_data_dir, f"{self.kgids[1]}_entity_id2uri.txt")

        with open(map1_fn) as file:
            lines = file.read().strip().split("\n")
            ent1_id_list = [line.split() for line in lines]
            ent1_id_list = [(int(newid), int(oldid)) for newid, oldid in ent1_id_list]
            ids_map1 = dict(ent1_id_list)
        with open(map2_fn) as file:
            lines = file.read().strip().split("\n")
            ent1_id_list = [line.split() for line in lines]
            ent1_id_list = [(int(newid), int(oldid)) for newid, oldid in ent1_id_list]
            ids_map2 = dict(ent1_id_list)

        # ent_to_uncert_map = {ids_map2[ids_map1[int(src_ent_ids[idx])]]: uncert_list[idx] for idx in range(len(src_ent_ids))}
        ent_to_uncert_map = {ids_map2[idx]: {"score1": score1_list[idx],
                                             "score5": score5_list[idx],
                                             "score10": score10_list[idx],
                                             "score15": score15_list[idx],
                                             "score20": score20_list[idx]} for idx in range(len(score1_list))}


        # ent_to_uncert_map = {e: ent_to_uncert_map[e] for e in target_entities}
        return ent_to_uncert_map


    def generate_msg(self):
        with open(os.path.join(self.subtask_data_dir, "kg_partition.json")) as file:
            obj = json.loads(file.read())
            g1_part_entities = obj["kg1_partition"]["entities"]
            exist_map = {str(e): True for e in g1_part_entities}

        pseudo_fn = os.path.join(self.subtask_data_dir, "pseudo_seeds.txt")
        if os.path.exists(pseudo_fn):
            existing_pseudo_seeds = read_alignment(pseudo_fn)
        else:
            existing_pseudo_seeds = []
        pseu_exist_map = {e1: True for e1, e2 in existing_pseudo_seeds}

        with open(os.path.join(self.subtask_data_dir, "new_pseudo_seeds_raw.txt")) as file:
            cont = file.read().strip()
            if cont == "":
                raw_pseudo_seeds = []
            else:
                raw_pseudo_seeds = read_alignment(os.path.join(self.subtask_data_dir, "new_pseudo_seeds_raw.txt"))
        uni_data = UniformData(self.subtask_data_dir, self.kgids)
        pseudo_seeds = [(uni_data.kg1_ent_id2uri_map[e1], uni_data.kg2_ent_id2uri_map[e2]) for e1, e2 in raw_pseudo_seeds]
        pseudo_seeds = [(e1, e2) for e1, e2 in pseudo_seeds if e1 in exist_map and e1 not in pseu_exist_map]

        pseudo_seeds = existing_pseudo_seeds + pseudo_seeds
        write_tab_lines(pseudo_seeds, pseudo_fn)

    def load_pseudo_seeds(self):
        pseudo_fn = os.path.join(self.subtask_data_dir, "pseudo_seeds.txt")
        if os.path.exists(pseudo_fn):
            alignment = read_alignment(pseudo_fn)
        else:
            alignment = []
        return alignment



class ContextBuilder(object):
    def __init__(self, data: UniformData, data_dir, kg_ids, part_n, out_dir=None, g1_ctx_only_once=True):
        self.data_dir = data_dir
        self.kgids = kg_ids
        # self.data = UniformData(self.data_dir, kg_ids)
        self.data = data
        self.out_dir = out_dir
        self.part_n = part_n
        self.g1_ctx_only_once = g1_ctx_only_once
        self.g1_ctx_have_done = False

    def build_contexts(self):
        for idx in range(self.part_n):
            part_idx = idx + 1
            g1_ctx, g2_ctx = self._build_context(part_idx)
            self._save_context(part_idx, g1_ctx, g2_ctx)

    def _build_context(self, part_idx):
        return [], []

    def build_g1_context(self):
        if self.g1_ctx_only_once and self.g1_ctx_have_done:
            return
        for idx in trange(self.part_n, desc="build g1 ctx"):
            part_idx = idx + 1
            self._build_g1_context(part_idx)
        self.g1_ctx_have_done = True

    def build_g2_context(self):
        for idx in trange(self.part_n, desc="build g2 ctx"):
            part_idx = idx + 1
            self._build_g2_context(part_idx)

    def _build_g1_context(self, part_idx):
        return []

    def _build_g2_context(self, part_idx):
        return []

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

    def get_neighbours2(self, triples, part_entities, max_hop_k):
        conn_arr = np.array(triples)[:, [0,2]]
        conn_df = pd.DataFrame(data=conn_arr, columns=["h", "t"])

        added_entity_set = set(part_entities)
        neighbours_list = []
        ent_df = pd.Series(data=part_entities).to_frame("ent")
        for step in range(0, max_hop_k):
            tmp_triple_df = ent_df.merge(conn_df, how="inner", left_on="ent", right_on="h")
            tmp_triple_df2 = ent_df.merge(conn_df, how="inner", left_on="ent", right_on="t")
            new_hop_entities = set(tmp_triple_df["t"].tolist()).difference(added_entity_set)
            new_hop_entities2 = set(tmp_triple_df2["h"].tolist()).difference(added_entity_set)
            new_hop_entities.update(new_hop_entities2)
            added_entity_set.update(new_hop_entities)
            if len(new_hop_entities) == 0:
                break
            neighbours_list.append(new_hop_entities)
            ent_df = pd.Series(data=list(new_hop_entities)).to_frame("ent")
        return neighbours_list

    def get_neighbours4(self, triples, part_entities, anchors, max_hop_k):
        if not hasattr(self, "conn_df"):
            conn_arr = np.array(triples)[:, [0,2]]
            conn_df = pd.DataFrame(data=conn_arr, columns=["h", "t"])
            self.conn_df = conn_df
        else:
            conn_df = self.conn_df

        added_entity_set = set(part_entities)
        ent_df = pd.Series(data=part_entities).to_frame("ent")
        anc_df = pd.Series(data=anchors).to_frame("anchor")

        real_added_nei_set = set()
        anchor_set = set(anchors)
        for step in range(0, max_hop_k):
            if step>0 and step == max_hop_k-1:
                anc_conn_df = anc_df.merge(conn_df, how="inner", left_on="anchor", right_on="t")
                tmp_triple_df = ent_df.merge(anc_conn_df, how="inner", left_on="ent", right_on="h")[["h", "t"]]
                anc_conn_df = anc_df.merge(conn_df, how="inner", left_on="anchor", right_on="h")
                tmp_triple_df2 = ent_df.merge(anc_conn_df, how="inner", left_on="ent", right_on="t")[["h", "t"]]
                final_added_set = set(tmp_triple_df.values.reshape((-1)).tolist()+tmp_triple_df2.values.reshape((-1)).tolist()).difference(added_entity_set)
                real_added_nei_set.update(final_added_set)
            else:
                tmp_triple_df = ent_df.merge(conn_df, how="inner", left_on="ent", right_on="h")
                tmp_triple_df2 = ent_df.merge(conn_df, how="inner", left_on="ent", right_on="t")
                new_hop_entities = set(tmp_triple_df["t"].tolist()).difference(added_entity_set)
                new_hop_entities2 = set(tmp_triple_df2["h"].tolist()).difference(added_entity_set)
                new_hop_entities.update(new_hop_entities2)
                added_entity_set.update(new_hop_entities)
                if len(new_hop_entities) == 0:
                    break
                real_added_nei_set.update(anchor_set.intersection(new_hop_entities))
                ent_df = pd.Series(data=list(new_hop_entities)).to_frame("ent")
        return real_added_nei_set

    def get_linking_entities_from_anchor_to_unmatch(self, triples, anchors, unmatched_entities, max_hop_k):
        conn_arr = np.array(triples)[:, [0,2]]
        conn_arr = np.concatenate([conn_arr, conn_arr[:, [1,0]]], axis=0)

        inpath_entities = set(anchors+unmatched_entities)
        added_entity_set = set(anchors)
        path_df = pd.Series(data=anchors).to_frame("t_0")
        linkent2hop_map = dict()
        for step in range(1, max_hop_k+1):
            conn_df = pd.DataFrame(data=conn_arr, columns=[f"h_{step}", f"t_{step}"])
            path_df = path_df.merge(conn_df, how="inner", left_on=f"t_{step-1}", right_on=f"h_{step}")

            new_hop_entities = set(path_df[f"t_{step}"].tolist()).difference(added_entity_set)
            if len(new_hop_entities) == 0:
                break
            path_df = pd.merge(pd.Series(list(new_hop_entities)).to_frame("cond"), path_df, how="inner", left_on="cond", right_on=f"t_{step}")
            col_names = [f"t_{i}" for i in range(step+1)]
            path_df = path_df[col_names]

            if step > 1:
                sub_path_df = pd.merge(pd.Series(list(unmatched_entities)).to_frame("unmat"), path_df, how="inner", left_on="unmat", right_on=f"t_{step}")
                for tmp_i in range(1, step):
                    tmp_link_entities = set(sub_path_df[f"t_{tmp_i}"].tolist()).difference(inpath_entities)
                    for e in tmp_link_entities:
                        linkent2hop_map[e] = step
                    inpath_entities.update(tmp_link_entities)

            added_entity_set.update(new_hop_entities)
        return linkent2hop_map


    def load_pseudo_seeds(self):
        seed_fn = os.path.join(self.data_dir, "all_pseudo_seeds.txt")
        if os.path.exists(seed_fn):
            pseudo_seeds = read_alignment(seed_fn)
        else:
            pseudo_seeds = []
        pseudo_seeds = list(set(pseudo_seeds))
        return pseudo_seeds


class G1Partitioner():
    def __init__(self, data:UniformData, data_dir, kgids, part_n):
        self.data_dir = data_dir
        self.part_n = part_n
        self.kgids = kgids
        # self.data = UniformData(data_dir=data_dir, kgids=kgids)
        self.data = data

    @abc.abstractmethod
    def _divide_entities(self):
        return []

    def partition_g1_entities(self):
        part_entities_list = self._divide_entities()
        for idx in range(len(part_entities_list)):
            part_data_dir = os.path.join(self.data_dir, f"partition_{idx+1}")
            part_entities = part_entities_list[idx]
            # part_triples = subgraph(part_entities, self.data.kg1_triples)
            part_triples = self.data.kg1_sub_triples(part_entities)
            train_alignment = self.data.load_train_alignment()
            part_train_align = sub_alignment_with_head(part_entities, train_alignment)
            obj = {
                "kg1_partition": {
                    "entities": part_entities_list[idx],
                    "triples": part_triples,
                    "train_alignment": part_train_align
                }
            }
            with open(os.path.join(part_data_dir, "kg_partition.json"), "w+") as file:
                file.write(json.dumps(obj))
        return part_entities_list


    def build_g1_subgraphs(self):
        for idx in trange(self.part_n, desc="build g1 subgraphs"):
            part_idx = idx+1
            self._build_g1_subgraph(part_idx)

    def _build_g1_subgraph(self, part_idx):
        part_data_dir = os.path.join(self.data_dir, f"partition_{part_idx}")
        with open(os.path.join(part_data_dir, "kg_partition.json")) as file:
            part_obj = json.loads(file.read())

        part_entities = part_obj["kg1_partition"]["entities"]
        context_entities = part_obj["kg1_partition"]["ctx_entities"]
        context_entities = list(set(context_entities).difference(set(part_entities)))

        part_data_dir = os.path.join(self.data_dir, f"partition_{part_idx}")
        if part_idx == 1:
            self.alignment_cache = self.data.load_all_alignment()
            self.train_alignment_cache = self.data.load_train_alignment()
            self.test_alignment_cache = self.data.load_test_alignment()
        alignment = self.alignment_cache
        train_alignment = self.train_alignment_cache
        test_alignment = self.test_alignment_cache

        # part_triples = subgraph(part_entities, self.data.kg1_triples)
        part_triples = self.data.kg1_sub_triples(part_entities + context_entities)
        # part_triples = [tuple(tri) for tri in part_triples]
        # if len(context_entities) > 0:
        #     # ctx_triples = subgraph(part_entities + context_entities, self.data.kg1_triples)
        #     ctx_triples = self.data.kg1_sub_triples(part_entities + context_entities)
        #     ctx_triples = [tuple(tri) for tri in ctx_triples]
        #     ctx_triples = list(set(ctx_triples).difference(part_triples))
        # else:
        #     ctx_triples = []
        part_align = sub_alignment_with_head(part_entities+context_entities, alignment)
        part_train_align = sub_alignment_with_head(part_entities+context_entities, train_alignment)
        part_test_align = sub_alignment_with_head(part_entities, test_alignment)
        ctx_train_align = sub_alignment_with_head(context_entities, train_alignment)
        kg1_part_obj = {  # after doing partG1, when is left in the subG1
            "entities": part_entities,
            "ctx_entities": context_entities,
            "triples": part_triples,
            # "ctx_triples": [], #ctx_triples,
            "alignment": part_align,  # original alignment located in this partition (including those involving ctx_entities)
            "train_alignment": part_train_align, # original train alignment located in this partition (not consider ctx_entities)
            "test_alignment": part_test_align,  # original test alignment located in this partition (not consider ctx_entities)
            # "ctx_train_alignment": [] #ctx_train_align  # pseudo train alignment
        }
        # save
        # part_data_dir = os.path.join(self.data_dir, f"partition_{i + 1}")
        if not os.path.exists(part_data_dir):
            os.mkdir(part_data_dir)
        with open(os.path.join(part_data_dir, "kg_partition.json"), "w+") as file:
            part_obj["kg1_partition"] = kg1_part_obj
            file.write(json.dumps(part_obj))


class G2Partitioner():
    def __init__(self, data:UniformData, data_dir, kgids, part_n, ctx_builder: ContextBuilder=None):
        self.data_dir = data_dir
        self.part_n = part_n
        self.kgids = kgids
        # self.data = UniformData(data_dir=data_dir, kgids=kgids)
        self.data = data
        self.ctx_builder = ctx_builder

    def load_pseudo_seeds(self):
        seed_fn = os.path.join(self.data_dir, "all_pseudo_seeds.txt")
        if os.path.exists(seed_fn):
            pseudo_seeds = read_alignment(seed_fn)
        else:
            pseudo_seeds = []
        pseudo_seeds = list(set(pseudo_seeds))
        return pseudo_seeds

    def get_anchors2(self, part_idx):
        train_alignment = self.data.load_train_alignment()
        pseudo_alignment = self.load_pseudo_seeds()
        all_train_alignment = train_alignment + pseudo_alignment

        with open(os.path.join(self.data_dir, f"partition_{part_idx}/tmp_part.json")) as file:
            part_entities = json.loads(file.read())
        ent2exist_map = {ent: True for ent in part_entities}
        anchors = set([e2 for e1, e2 in all_train_alignment if e1 in ent2exist_map])
        return anchors

    @abc.abstractmethod
    def _select_g2_candidates(self, part_idx):
        pass

    # @staticmethod
    # def _subgraph(part_entities, all_triples):
    #     ent2bool_map = {e: True for e in part_entities}
    #     bucket = []
    #     for h, r, t in all_triples:
    #         if ent2bool_map.get(h, False) and ent2bool_map.get(t, False):
    #             bucket.append((h, r, t))
    #     return bucket

    def partition_g2_entities(self):
        for idx in trange(self.part_n, desc="partition g2"):
            part_idx = idx + 1
            part_entities = self._select_g2_candidates(part_idx)
            # part_triples = subgraph(part_entities, self.data.kg2_triples)
            # part_triples = self.data.kg2_sub_triples(part_entities)
            part_data_dir = os.path.join(self.data_dir, f"partition_{part_idx}")
            with open(os.path.join(part_data_dir, "kg_partition.json")) as file:
                obj = json.loads(file.read())
            obj["kg2_partition"] = {
                "entities": part_entities,
                # "triples": part_triples
            }
            with open(os.path.join(part_data_dir, "kg_partition.json"), "w+") as file:
                file.write(json.dumps(obj))

    def build_g2_subgraphs(self):
        for idx in trange(self.part_n, desc="build g2 subgraphs"):
            part_idx = idx + 1
            self._build_g2_subgraph(part_idx)

    def _build_g2_subgraph(self, part_idx):
        part_data_dir = os.path.join(self.data_dir, f"partition_{part_idx}")

        with open(os.path.join(part_data_dir, "kg_partition.json")) as file:
            part_obj = json.loads(file.read())
            part_entities = []
            part_entities.extend(part_obj["kg2_partition"]["entities"])
            if "ctx_entities" in part_obj["kg2_partition"]:
                ctx_entities = part_obj["kg2_partition"]["ctx_entities"]
                part_entities.extend(ctx_entities)
                part_entities = list(set(part_entities))

        kg2_triples = self.data.kg2_triples
        with open(os.path.join(part_data_dir, "kg_partition.json")) as file:
            kg1_obj = json.loads(file.read())["kg1_partition"]
            alignment = kg1_obj["alignment"]
            train_alignment = kg1_obj["train_alignment"]
            test_alignment = kg1_obj["test_alignment"]
            # ctx_train_alignment = kg1_obj["ctx_train_alignment"]

        # def _subgraph(part_entities, all_triples):
        #     ent2bool_map = {e: True for e in part_entities}
        #     bucket = []
        #     for h, r, t in all_triples:
        #         if ent2bool_map.get(h, False) and ent2bool_map.get(t, False):
        #             bucket.append((h, r, t))
        #     return bucket
        #
        # def _sub_alignment(part_entities, alignment):
        #     ent2bool_map = {e: True for e in part_entities}
        #     bucket = []
        #     for e1, e2 in alignment:
        #         if ent2bool_map.get(e2, False):
        #             bucket.append((e1, e2))
        #     return bucket

        # collect info for each partition
        # part_triples = subgraph(part_entities, kg2_triples)
        part_triples = self.data.kg2_sub_triples(part_entities)
        part_align = sub_alignment_with_tail(part_entities, alignment)
        part_train_align = sub_alignment_with_tail(part_entities, train_alignment)
        part_test_align = sub_alignment_with_tail(part_entities, test_alignment)
        # part_ctx_train_align = sub_alignment_with_tail(part_entities, ctx_train_alignment)
        # kg2_partition = (ent_part, part_triples, part_align, part_train_align, part_test_align)
        g2_part_obj = {
            "entities": part_entities,
            "ctx_entities": part_obj["kg2_partition"]["ctx_entities"],
            "triples": part_triples,
            "alignment": part_align,
            "train_alignment": part_train_align,
            "test_alignment": part_test_align,
            # "ctx_train_alignment": [] #part_ctx_train_align
        }

        # save partitioning result
        part_obj["kg2_partition"] = g2_part_obj
        with open(os.path.join(part_data_dir, "kg_partition.json"), "w+") as file:
            file.write(json.dumps(part_obj))

        return part_obj



class Server:
    def __init__(self, data_dir, kgids, out_dir, part_n, g1_partitioner: G1Partitioner, g2_partitioner: G2Partitioner, ctx_builder: ContextBuilder=None):
        self.g1_partitioner = g1_partitioner
        self.g2_partitioner = g2_partitioner
        # self.context_builder = context_builder
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.part_n = part_n
        self.kgids = kgids
        self.ctx_builder = ctx_builder

    def communicate(self):
        all_pseudo_seeds = []
        for idx in range(self.part_n):
            part_idx = idx + 1
            pseudo_fn = os.path.join(self.data_dir, f"partition_{part_idx}/pseudo_seeds.txt")
            with open(pseudo_fn) as file:
                cont = file.read().strip()
            if cont == "":
                part_pseudo_seeds = []
            else:
                part_pseudo_seeds = read_alignment(pseudo_fn)
            all_pseudo_seeds.extend(part_pseudo_seeds)
        all_pseudo_seeds = list(set(all_pseudo_seeds))
        write_tab_lines(all_pseudo_seeds, os.path.join(self.data_dir, "all_pseudo_seeds.txt"))


    def check_stop(self):
        pass







