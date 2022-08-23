# -*- coding: utf-8 -*-

import os
import argparse


def transform_openea_to_uniform(data_dir, kgids, out_dir):
    kg1_id, kg2_id = kgids

    ent1_list_inlinks = []
    ent2_list_inlinks = []
    with open(os.path.join(data_dir, "ent_links")) as file:
        lines = file.read().strip().split("\n")
        for line in lines:
            ent1, ent2 = line.split("\t")
            ent1_list_inlinks.append(ent1)
            ent2_list_inlinks.append(ent2)

    with open(os.path.join(data_dir, "rel_triples_1")) as file:
        lines = file.read().strip().split("\n")
        ent1_list = []
        rel1_list = []
        for line in lines:
            h,r,t = line.split("\t")
            rel1_list.append(r)
            ent1_list.append(h)
            ent1_list.append(t)
        ent1_list = sorted(list(set(ent1_list+ent1_list_inlinks)))
        ent1_oldid2newid_map = dict()
        ent1_lines = []
        for idx, entid in enumerate(ent1_list):
            ent1_oldid2newid_map[entid] = idx
            ent1_lines.append(f"{idx}\t{entid}")

        rel1_list = sorted(list(set(rel1_list)))
        rel1_oldid2newid_map = dict()
        rel1_lines = []
        for idx, relid in enumerate(rel1_list):
            rel1_oldid2newid_map[relid] = idx
            rel1_lines.append(f"{idx}\t{relid}")
        triple1_new_lines = []
        for line in lines:
            h, r, t = line.split("\t")
            triple1_new_lines.append(f"{ent1_oldid2newid_map[h]}\t{rel1_oldid2newid_map[r]}\t{ent1_oldid2newid_map[t]}")
    with open(os.path.join(out_dir, f"{kg1_id}_triple_rel.txt"), "w+") as file:
        new_cont = "\n".join(triple1_new_lines)
        file.write(new_cont)
    with open(os.path.join(out_dir, f"{kg1_id}_relation_id2uri.txt"), "w+") as file:
        file.write("\n".join(rel1_lines))
    with open(os.path.join(out_dir, f"{kg1_id}_entity_id2uri.txt"), "w+") as file:
        new_cont = "\n".join(ent1_lines)
        file.write(new_cont)


    with open(os.path.join(data_dir, "rel_triples_2")) as file:
        lines = file.read().strip().split("\n")
        ent2_list = []
        rel2_list = []
        for line in lines:
            h,r,t = line.split("\t")
            rel2_list.append(r)
            ent2_list.append(h)
            ent2_list.append(t)
        ent2_list = sorted(list(set(ent2_list+ent2_list_inlinks)))
        ent2_oldid2newid_map = dict()
        ent2_lines = []
        for idx, entid in enumerate(ent2_list):
            ent2_oldid2newid_map[entid] = idx
            ent2_lines.append(f"{idx}\t{entid}")

        rel2_list = sorted(list(set(rel2_list)))
        rel2_oldid2newid_map = dict()
        rel2_lines = []
        for idx, relid in enumerate(rel2_list):
            rel2_oldid2newid_map[relid] = idx
            rel2_lines.append(f"{idx}\t{relid}")
        triple2_new_lines = []
        for line in lines:
            h, r, t = line.split("\t")
            triple2_new_lines.append(f"{ent2_oldid2newid_map[h]}\t{rel2_oldid2newid_map[r]}\t{ent2_oldid2newid_map[t]}")
        new_cont = "\n".join(triple2_new_lines)
    with open(os.path.join(out_dir, f"{kg2_id}_triple_rel.txt"), "w+") as file:
        file.write(new_cont)
    with open(os.path.join(out_dir, f"{kg2_id}_relation_id2uri.txt"), "w+") as file:
        file.write("\n".join(rel2_lines))
    with open(os.path.join(out_dir, f"{kg2_id}_entity_id2uri.txt"), "w+") as file:
        new_cont = "\n".join(ent1_lines)
        file.write(new_cont)

    with open(os.path.join(data_dir, "ent_links")) as file:
        lines = file.read().strip().split("\n")
        align_lines = []
        for line in lines:
            ent1, ent2 = line.split("\t")
            new_line = f"{ent1_oldid2newid_map[ent1]}\t{ent2_oldid2newid_map[ent2]}"
            align_lines.append(new_line)

    with open(os.path.join(out_dir, "alignment_of_entity.txt"), "w+") as file:
        file.write("\n".join(align_lines))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--kgids', type=str)
    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    transform_openea_to_uniform(args.data_dir, args.kgids.split(","), args.out_dir)



