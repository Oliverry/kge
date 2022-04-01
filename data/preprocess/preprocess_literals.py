#!/usr/bin/env python
"""
Preprocess a given literals files into a suitable form for LibKGE.
The literals file has to be in the corresponding folder named literals.txt
such that we have the tab separated columns entity_ids, relations, literals.
"""

import argparse
import os
import sys
from os.path import exists

import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    args = parser.parse_args()

    literal_file_path = os.path.join(args.folder, "literals.txt")
    if exists(literal_file_path):
        print("Found literals file " + literal_file_path)
        print("Starting preprocessing")
        # dicts for processing
        entity_ids = {}
        relation_ids = {}

        # create mapping from freebase ids to internal indexes
        entity_file_path = os.path.join(args.folder, "entity_ids.del")
        entity_file = open(entity_file_path, "r")
        for line in entity_file:
            if len(line) > 0:
                line_tmp = line.replace("\n", "")
                line_parts = line_tmp.split("\t")
                entity_ids[line_parts[1]] = line_parts[0]

        # create mapping from freebase relation urls to indexes
        literal_file = open(literal_file_path, "r")
        for line in literal_file:
            if len(line) > 0:
                line_tmp = line.replace("\n", "")
                line_parts = line_tmp.split("\t")
                if not line_parts[1] in relation_ids:
                    relation_ids[line_parts[1]] = len(relation_ids)

        # create new numerical triples and save them
        literal_del_file_path = os.path.join(args.folder, "literals.del")
        literal_del_file = open(literal_del_file_path, "w")
        literal_file.seek(0)
        num_literals = 0
        for line in literal_file:
            if len(line) > 0:
                num_literals += 1
                line_tmp = line.replace("\n", "")
                line_parts = line_tmp.split("\t")
                # if entity is contained in dataset
                if line_parts[0] in entity_ids:
                    literal_del_file.write(
                        str(entity_ids[line_parts[0]]) + "\t" +
                        str(relation_ids[line_parts[1]]) + "\t" +
                        str(float(line_parts[2])) + "\n"
                    )
        literal_del_file.close()

        # write information to dataset yaml file
        config_file_path = os.path.join(args.folder, "dataset.yaml")
        with open(config_file_path, "r+") as file:
            try:
                config = yaml.safe_load(file)
                config["dataset"]["files.literals.filename"] = "literals.del"
                config["dataset"]["files.literals.type"] = "triples"
                config["dataset"]["files.literals.size"] = num_literals
                file.seek(0)
                file.truncate()
                yaml.dump(config, file)
            except yaml.YAMLError as exception:
                print(exception)
                sys.exit(0)
    else:
        print("Found no literals for dataset.")
