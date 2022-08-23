# -*- coding: utf-8 -*-

import os
import random
import networkx as nx
import numpy as np


def read_tab_lines(fn):
    with open(fn) as file:
        cont = file.read().strip()
        if cont == "":
            return []
        lines = cont.split("\n")
        tuple_list = []
        for line in lines:
            t = line.split("\t")
            tuple_list.append(t)
    return tuple_list


def write_tab_lines(tuple_list, fn):
    with open(fn, "w+") as file:
        for tup in tuple_list:
            s_tup = [str(e) for e in tup]
            file.write("\t".join(s_tup) + "\n")


def read_alignment(fn):
    alignment = read_tab_lines(fn)
    alignment = [(int(ent1_id), int(ent2_id)) for ent1_id, ent2_id in alignment]
    return alignment


class UniformData:
    def __init__(self, data_dir, kgids=None):
        self.data_dir = data_dir
        data_name = os.path.dirname(data_dir)
        if kgids:
            kgid1, kgid2 = kgids
        else:
            kgid1, kgid2 = data_name.split("_")

        kg1_ent_id_uri_list, kg1_rel_id_uri_list, self.kg1_triples = self.load_kg(kgid1)
        self.kg1_ent_id2uri_map = dict(kg1_ent_id_uri_list)
        self.kg1_rel_id2uri_map = dict(kg1_rel_id_uri_list)

        kg2_ent_id_uri_list, kg2_rel_id_uri_list, self.kg2_triples = self.load_kg(kgid2)
        self.kg2_ent_id2uri_map = dict(kg2_ent_id_uri_list)
        self.kg2_rel_id2uri_map = dict(kg2_rel_id_uri_list)

        self.kg1_entities = sorted(list(self.kg1_ent_id2uri_map.keys()))
        self.kg2_entities = sorted(list(self.kg2_ent_id2uri_map.keys()))
        self.kg1_relations = sorted(list(self.kg1_rel_id2uri_map.keys()))
        self.kg2_relations = sorted(list(self.kg2_rel_id2uri_map.keys()))

        #
        self.kg1_triple_arr = np.array(self.kg1_triples)
        self.kg1_head2triples_map = {}
        self.kg1_tail2triples_map = {}
        for idx, (h,r,t) in enumerate(self.kg1_triples):
            if h not in self.kg1_head2triples_map:
                self.kg1_head2triples_map[h] = []
            self.kg1_head2triples_map[h].append(idx)
            if t not in self.kg1_tail2triples_map:
                self.kg1_tail2triples_map[t] = []
            self.kg1_tail2triples_map[t].append(idx)
        self.kg2_triple_arr = np.array(self.kg2_triples)
        self.kg2_head2triples_map = {}
        self.kg2_tail2triples_map = {}
        for idx, (h,r,t) in enumerate(self.kg2_triples):
            if h not in self.kg2_head2triples_map:
                self.kg2_head2triples_map[h] = []
            self.kg2_head2triples_map[h].append(idx)
            if t not in self.kg2_tail2triples_map:
                self.kg2_tail2triples_map[t] = []
            self.kg2_tail2triples_map[t].append(idx)

    def load_kg(self, kgid):
        ent_id_uri_list = read_tab_lines(os.path.join(self.data_dir, f"{kgid}_entity_id2uri.txt"))
        rel_id_uri_list = read_tab_lines(os.path.join(self.data_dir, f"{kgid}_relation_id2uri.txt"))
        triple_rel_list = read_tab_lines(os.path.join(self.data_dir, f"{kgid}_triple_rel.txt"))

        ent_id_uri_list = [(int(id), uri) for id, uri in ent_id_uri_list]
        rel_id_uri_list = [(int(id), uri) for id, uri in rel_id_uri_list]
        triple_rel_list = [(int(ent1_id), int(ent2_id), int(rel_id)) for ent1_id, ent2_id, rel_id in triple_rel_list]
        return ent_id_uri_list, rel_id_uri_list, triple_rel_list

    def load_all_alignment(self):
        alignment = read_tab_lines(os.path.join(self.data_dir, "alignment_of_entity.txt"))
        alignment = [(int(ent1_id), int(ent2_id)) for ent1_id, ent2_id in alignment]
        return alignment

    def load_train_alignment(self):
        alignment = read_tab_lines(os.path.join(self.data_dir, "train_alignment.txt"))
        alignment = [(int(ent1_id), int(ent2_id)) for ent1_id, ent2_id in alignment]
        return alignment

    def load_test_alignment(self):
        alignment = read_tab_lines(os.path.join(self.data_dir, "test_alignment.txt"))
        alignment = [(int(ent1_id), int(ent2_id)) for ent1_id, ent2_id in alignment]
        return alignment

    def kg1_sub_triples(self, entities):
        head2idxes = []
        tail2idxes = []
        for e in entities:
            head2idxes.extend(self.kg1_head2triples_map.get(e, []))
            tail2idxes.extend(self.kg1_tail2triples_map.get(e, []))
        inter_idxes = list(set(head2idxes).intersection(set(tail2idxes)))
        sub_triples = self.kg1_triple_arr[inter_idxes].tolist()
        return sub_triples

    def kg2_sub_triples(self, entities):
        head2idxes = []
        tail2idxes = []
        for e in entities:
            head2idxes.extend(self.kg2_head2triples_map.get(e, []))
            tail2idxes.extend(self.kg2_tail2triples_map.get(e, []))
        inter_idxes = list(set(head2idxes).intersection(set(tail2idxes)))
        sub_triples = self.kg2_triple_arr[inter_idxes].tolist()
        return sub_triples

    def divide_train_test(self, train_percent):
        all_alignment = self.load_all_alignment()
        num = len(all_alignment)
        train_num = int(num * train_percent)
        random.shuffle(all_alignment)
        train_alignment = all_alignment[:train_num]
        test_alignment = all_alignment[train_num:]
        pseudo_fn = os.path.join(self.data_dir, "name_pseudo_mappings.txt")
        if os.path.exists(pseudo_fn):
            pseudo_mappings = read_tab_lines(pseudo_fn)
            pseudo_mappings = [(int(ent1_id), int(ent2_id)) for ent1_id, ent2_id in pseudo_mappings]
            train_alignment = train_alignment + pseudo_mappings
        write_tab_lines(train_alignment, os.path.join(self.data_dir, "train_alignment.txt"))
        write_tab_lines(test_alignment, os.path.join(self.data_dir, "test_alignment.txt"))



def convert_uniform_to_rrea(data_dir, kgids):
    uni_data = UniformData(data_dir, kgids)

    kg1_entities = uni_data.kg1_entities
    kg2_entities = uni_data.kg2_entities
    kg1_ent_uri_tuples = list(enumerate(kg1_entities))
    kg1_ent_num = len(kg1_entities)
    kg2_ent_uri_tuples = [(idx+kg1_ent_num, ent2) for idx, ent2 in enumerate(kg2_entities)]

    kg1_ent_old2new_id_map = {oldid: newid for newid, oldid in kg1_ent_uri_tuples}
    kg2_ent_old2new_id_map = {oldid: newid for newid, oldid in kg2_ent_uri_tuples}

    kg1_relations = uni_data.kg1_relations
    kg2_relations = uni_data.kg2_relations
    kg1_rel_uri_tuples = list(enumerate(kg1_relations))
    kg1_rel_num = len(kg2_relations)
    kg2_rel_uri_tuples = [(idx + kg1_rel_num, rel2) for idx, rel2 in enumerate(kg2_relations)]

    kg1_rel_old2new_id_map = {oldid: newid for newid, oldid in kg1_rel_uri_tuples}
    kg2_rel_old2new_id_map = {oldid: newid for newid, oldid in kg2_rel_uri_tuples}


    new_kg1_triples = [(kg1_ent_old2new_id_map[h], kg1_rel_old2new_id_map[r], kg1_ent_old2new_id_map[t]) for h,r,t in uni_data.kg1_triples]
    new_kg2_triples = [(kg2_ent_old2new_id_map[h], kg2_rel_old2new_id_map[r], kg2_ent_old2new_id_map[t]) for h, r, t in uni_data.kg2_triples]

    new_all_alignment = [(kg1_ent_old2new_id_map[e1], kg2_ent_old2new_id_map[e2]) for e1, e2 in uni_data.load_all_alignment()]
    new_train_alignment = [(kg1_ent_old2new_id_map[e1], kg2_ent_old2new_id_map[e2]) for e1, e2 in uni_data.load_train_alignment()]
    test_alignment = uni_data.load_test_alignment()
    valid_test_alignment = []
    invalid_test_alignment = []
    new_test_alignment = []
    for e1, e2 in test_alignment:
        if e2 in kg2_ent_old2new_id_map:
            new_test_alignment.append((kg1_ent_old2new_id_map[e1], kg2_ent_old2new_id_map[e2]))
            valid_test_alignment.append((e1, e2))
        else:
            invalid_test_alignment.append((e1, e2))
    # new_test_alignment = [(kg1_ent_old2new_id_map[e1], kg2_ent_old2new_id_map[e2]) for e1, e2 in uni_data.load_test_alignment()]

    write_tab_lines(kg1_ent_uri_tuples, os.path.join(data_dir, "ent_ids_1"))
    write_tab_lines(kg2_ent_uri_tuples, os.path.join(data_dir, "ent_ids_2"))
    write_tab_lines(new_kg1_triples, os.path.join(data_dir, "triples_1"))
    write_tab_lines(new_kg2_triples, os.path.join(data_dir, "triples_2"))
    write_tab_lines(new_all_alignment, os.path.join(data_dir, "ref_ent_ids"))
    write_tab_lines(new_train_alignment, os.path.join(data_dir, "ref_ent_ids_train"))
    write_tab_lines(new_test_alignment, os.path.join(data_dir, "ref_ent_ids_test"))
    write_tab_lines(invalid_test_alignment, os.path.join(data_dir, "test_alignment_invalid.txt"))
    write_tab_lines(valid_test_alignment, os.path.join(data_dir, "test_alignment_valid.txt"))



def convert_uniform_to_openea(data_dir, kgids, out_dir=None):
    uni_data = UniformData(data_dir, kgids)

    ent_links = [(uni_data.kg1_ent_id2uri_map[e1], uni_data.kg2_ent_id2uri_map[e2]) for e1, e2 in uni_data.load_all_alignment()]
    kg1_rel_triples = [(uni_data.kg1_ent_id2uri_map[h], uni_data.kg1_rel_id2uri_map[r], uni_data.kg1_ent_id2uri_map[t]) for h,r,t in uni_data.kg1_triples]
    kg2_rel_triples = [(uni_data.kg2_ent_id2uri_map[h], uni_data.kg2_rel_id2uri_map[r], uni_data.kg2_ent_id2uri_map[t]) for h,r,t in uni_data.kg2_triples]

    train_ent_links = [(uni_data.kg1_ent_id2uri_map[e1], uni_data.kg2_ent_id2uri_map[e2]) for e1, e2 in uni_data.load_train_alignment()]
    test_alignment = uni_data.load_test_alignment()
    valid_test_alignment = []
    invalid_test_alignment = []
    new_test_alignment = []
    for e1, e2 in test_alignment:
        if e2 in uni_data.kg2_ent_id2uri_map:
            new_test_alignment.append((uni_data.kg1_ent_id2uri_map[e1], uni_data.kg2_ent_id2uri_map[e2]))
            valid_test_alignment.append((e1, e2))
        else:
            invalid_test_alignment.append((e1, e2))
    if out_dir is None:
        out_dir = os.path.join(data_dir, "openea_format")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    write_tab_lines(kg1_rel_triples, os.path.join(out_dir, "rel_triples_1"))
    write_tab_lines(kg2_rel_triples, os.path.join(out_dir, "rel_triples_2"))
    write_tab_lines(ent_links, os.path.join(out_dir, "ent_links"))
    with open(os.path.join(out_dir, "attr_triples_1"), "w+") as file:
        file.write("")
    with open(os.path.join(out_dir, "attr_triples_2"), "w+") as file:
        file.write("")


    partition_dir = os.path.join(out_dir, "partition")
    if not os.path.exists(partition_dir):
        os.makedirs(partition_dir)

    write_tab_lines(train_ent_links, os.path.join(partition_dir, "train_links"))
    write_tab_lines(new_test_alignment, os.path.join(partition_dir, "test_links"))
    write_tab_lines(invalid_test_alignment, os.path.join(partition_dir, "test_alignment_invalid.txt"))
    write_tab_lines(valid_test_alignment, os.path.join(partition_dir, "test_alignment_valid.txt"))
    with open(os.path.join(partition_dir, "valid_links"), "w+") as file:
        cont = "\n".join([])
        file.write(cont)






