# -*- coding: utf-8 -*-


import os
from divea.data2 import UniformData, write_tab_lines
import argparse


def convert_uniform_to_openea(data_dir, kgids, out_dir=None):
    uni_data = UniformData(data_dir, kgids)

    ent_links = [(uni_data.kg1_ent_id2uri_map[e1], uni_data.kg2_ent_id2uri_map[e2]) for e1, e2 in uni_data.load_all_alignment()]
    kg1_rel_triples = [(uni_data.kg1_ent_id2uri_map[h], uni_data.kg1_rel_id2uri_map[r], uni_data.kg1_ent_id2uri_map[t]) for h,r,t in uni_data.kg1_triples]
    kg2_rel_triples = [(uni_data.kg2_ent_id2uri_map[h], uni_data.kg2_rel_id2uri_map[r], uni_data.kg2_ent_id2uri_map[t]) for h,r,t in uni_data.kg2_triples]

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


    # partition_dir = os.path.join(out_dir, "partition")
    # if not os.path.exists(partition_dir):
    #     os.makedirs(partition_dir)
    #
    #
    #
    # write_tab_lines(train_ent_links, os.path.join(partition_dir, "train_links"))
    # write_tab_lines(new_test_alignment, os.path.join(partition_dir, "test_links"))
    # write_tab_lines(invalid_test_alignment, os.path.join(partition_dir, "test_alignment_invalid.txt"))
    # write_tab_lines(valid_test_alignment, os.path.join(partition_dir, "test_alignment_valid.txt"))
    # with open(os.path.join(partition_dir, "valid_links"), "w+") as file:
    #     cont = "\n".join([])
    #     file.write(cont)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--kgids', type=str)
    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    convert_uniform_to_openea(args.data_dir, args.kgids.split(","), args.out_dir)