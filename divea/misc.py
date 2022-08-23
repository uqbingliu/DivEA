# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


# def subgraph(part_entities, all_triples):
#     ent2bool_map = {e: True for e in part_entities}
#     bucket = []
#     for h, r, t in all_triples:
#         if ent2bool_map.get(h, False) and ent2bool_map.get(t, False):
#             bucket.append((h, r, t))
#     return bucket

# def subgraph(part_entities, all_triples):
#     ent_df = pd.Series(data=part_entities).to_frame("ent")
#     triple_df = pd.DataFrame(data=all_triples, columns=["h", "r", "t"])
#     triple_df = ent_df.merge(triple_df, how="inner", left_on="ent", right_on="h")
#     triple_df = ent_df.merge(triple_df, how="inner", left_on="ent", right_on="t")
#     triple_df = triple_df[["h", "r", "t"]]
#     triple_list = triple_df.values.tolist()
#     return triple_list


def sub_alignment_with_head(part_entities, alignment):
    if not isinstance(part_entities, set):
        part_entities = set(part_entities)
    bucket = []
    for e1, e2 in alignment:
        if e1 in part_entities:
            bucket.append((e1, e2))
    return bucket

# def sub_alignment_with_head(part_entities, alignment):
#     ent_df = pd.Series(data=part_entities).to_frame("ent")
#     alignment_df = pd.DataFrame(data=alignment, columns=["e1", "e2"])
#     alignment_df = ent_df.merge(alignment_df, how="inner", left_on="ent", right_on="e1")
#     alignment_df = alignment_df[["e1", "e2"]]
#     alignment = alignment_df.values.tolist()
#     return alignment


def sub_alignment_with_tail(part_entities, alignment):
    if not isinstance(part_entities, set):
        part_entities = set(part_entities)
    bucket = []
    for e1, e2 in alignment:
        if e2 in part_entities:
            bucket.append((e1, e2))
    return bucket

# def sub_alignment_with_tail(part_entities, alignment):
#     ent_df = pd.Series(data=part_entities).to_frame("ent")
#     alignment_df = pd.DataFrame(data=alignment, columns=["e1", "e2"])
#     alignment_df = ent_df.merge(alignment_df, how="inner", left_on="ent", right_on="e2")
#     alignment_df = alignment_df[["e1", "e2"]]
#     alignment = alignment_df.values.tolist()
#     return alignment


def get_neighbours(conn_df, part_entities, max_hop_k):
    # conn_arr = np.array(triples)[:, [0,2]]
    # conn_df = pd.DataFrame(data=conn_arr, columns=["h", "t"])

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

