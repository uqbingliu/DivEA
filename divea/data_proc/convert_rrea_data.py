# -*- coding: utf-8 -*-

import os
import argparse


def transform_rrea_to_uniform(data_dir, kgids, out_dir):
    kg1_id, kg2_id = kgids

    ent1_oldid2newid_map = dict()
    with open(os.path.join(data_dir, "ent_ids_1")) as file:
        lines = file.read().strip().split("\n")
        ent1_new_lines = []
        for idx, line in enumerate(lines):
            old_ent1, _ = line.split()
            ent1_new_lines.append(f"{idx}\t{old_ent1}")
            ent1_oldid2newid_map[old_ent1] = str(idx)
        new_cont = "\n".join(ent1_new_lines)
    with open(os.path.join(out_dir, f"{kg1_id}_entity_id2uri.txt"), "w+") as file:
        file.write(new_cont)

    ent2_oldid2newid_map = dict()
    with open(os.path.join(data_dir, "ent_ids_2")) as file:
        lines = file.read().strip().split("\n")
        ent2_new_lines = []
        for idx, line in enumerate(lines):
            old_ent2, _ = line.split()
            ent2_new_lines.append(f"{idx}\t{old_ent2}")
            ent2_oldid2newid_map[old_ent2] = str(idx)
        new_cont = "\n".join(ent2_new_lines)
    with open(os.path.join(out_dir, f"{kg2_id}_entity_id2uri.txt"), "w+") as file:
        file.write(new_cont)

    with open(os.path.join(data_dir, "triples_1")) as file:
        lines = file.read().strip().split("\n")
        rel1_list = []
        for line in lines:
            h,r,t = line.split()
            rel1_list.append(int(r))
        rel1_list = sorted(list(set(rel1_list)))
        rel1_oldid2newid_map = dict()
        rel1_lines = []
        for idx, relid in enumerate(rel1_list):
            rel1_oldid2newid_map[relid] = idx
            rel1_lines.append(f"{idx}\t{relid}")
        triple1_new_lines = []
        for line in lines:
            h, r, t = line.split()
            triple1_new_lines.append(f"{ent1_oldid2newid_map[h]}\t{rel1_oldid2newid_map[int(r)]}\t{ent1_oldid2newid_map[t]}")
    with open(os.path.join(out_dir, f"{kg1_id}_triple_rel.txt"), "w+") as file:
        new_cont = "\n".join(triple1_new_lines)
        file.write(new_cont)
    with open(os.path.join(out_dir, f"{kg1_id}_relation_id2uri.txt"), "w+") as file:
        file.write("\n".join(rel1_lines))

    with open(os.path.join(data_dir, "triples_2")) as file:
        lines = file.read().strip().split("\n")
        rel2_list = []
        for line in lines:
            h,r,t = line.split()
            rel2_list.append(int(r))
        rel2_list = sorted(list(set(rel2_list)))
        rel2_oldid2newid_map = dict()
        rel2_lines = []
        for idx, relid in enumerate(rel2_list):
            rel2_oldid2newid_map[relid] = idx
            rel2_lines.append(f"{idx}\t{relid}")
        triple2_new_lines = []
        for line in lines:
            h, r, t = line.split()
            triple2_new_lines.append(f"{ent2_oldid2newid_map[h]}\t{rel2_oldid2newid_map[int(r)]}\t{ent2_oldid2newid_map[t]}")
        new_cont = "\n".join(triple2_new_lines)
    with open(os.path.join(out_dir, f"{kg2_id}_triple_rel.txt"), "w+") as file:
        file.write(new_cont)
    with open(os.path.join(out_dir, f"{kg2_id}_relation_id2uri.txt"), "w+") as file:
        file.write("\n".join(rel2_lines))

    with open(os.path.join(data_dir, "ref_ent_ids")) as file:
        lines = file.read().strip().split("\n")
        align_lines = []
        for line in lines:
            ent1, ent2 = line.split()
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
    if os.path.exists(os.path.join(args.data_dir, "ent_links")):
        from convert_openea_data import transform_openea_to_uniform
        transform_openea_to_uniform(args.data_dir, args.kgids.split(","), args.out_dir)
    else:
        transform_rrea_to_uniform(args.data_dir, args.kgids.split(","), args.out_dir)



