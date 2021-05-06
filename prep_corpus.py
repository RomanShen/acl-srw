#!/usr/bin/env python
# -*- coding:utf-8 -*-
""" 
@author: romanshen 
@file: prep_corpus.py 
@time: 2021/05/06
@contact: xiangqing.shen@njust.edu.cn
"""


import json
from collections import defaultdict
import os

import hydra
from omegaconf import OmegaConf


def run(cfg):
    os.chdir(hydra.utils.get_original_cwd())
    triggers = dict()
    rule_mappings = dict()

    pubmed = defaultdict(dict)
    with open(cfg.silver_data_json) as f:
        raw = json.load(f)
        for entry in raw:
            sentence = entry["sentence"]
            rule = entry["rule"]
            trigger = entry["trigger"]
            entity = entry["entity"]

            eid = "%s%i%i" % (entity[0], entity[1][0], entity[1][1])
            temp = {"events": [{"trigger": trigger, "rule": rule}], "entity": entity}
            pubmed[sentence][eid] = temp
    i = 0
    for sentence in pubmed:
        i += 1
        with open("%s/%d.txt" % (cfg.path_silver_processed_dir, i), "w") as txt:
            txt.write(sentence)
        j = 1
        k = len(pubmed[sentence].keys()) + 1
        l = 1
        for eid in pubmed[sentence]:
            entity = pubmed[sentence][eid]["entity"]
            with open("%s/%d.a1" % (cfg.path_silver_processed_dir, i), "a") as a1:
                a1.write(
                    "T%d\tProtein %d %d\t%s\n"
                    % (j, entity[1][0], entity[1][1], entity[0])
                )
            for event in pubmed[sentence][eid]["events"]:
                trigger = event["trigger"]
                rule = event["rule"]
                rule_mappings[
                    "%s/%d/E%d" % (cfg.path_silver_processed_dir, i, l)
                ] = rule
                with open("%s/%d.a2" % (cfg.path_silver_processed_dir, i), "a") as a2:
                    if (
                        "%s%d%d" % (trigger[0], trigger[1][0], trigger[1][1])
                        not in triggers
                    ):
                        triggers[
                            "%s%d%d" % (trigger[0], trigger[1][0], trigger[1][1])
                        ] = k
                        k += 1
                    a2.write(
                        "T%d\t%s %d %d\t%s\n"
                        % (
                            triggers[
                                "%s%d%d" % (trigger[0], trigger[1][0], trigger[1][1])
                            ],
                            cfg.event_type,
                            trigger[1][0],
                            trigger[1][1],
                            trigger[0],
                        )
                    )
                    a2.write(
                        "E%d\t%s:T%d Theme:T%d\n"
                        % (
                            l,
                            cfg.event_type,
                            triggers[
                                "%s%d%d" % (trigger[0], trigger[1][0], trigger[1][1])
                            ],
                            j,
                        )
                    )
                l += 1
            j += 1

    with open("%s/rule_mappings.json" % cfg.path_silver_processed_dir, "w") as f:
        f.write(json.dumps(rule_mappings))


@hydra.main(config_path="conf", config_name="conf_ge_corpus")
def run_prep(cfg):
    print(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    run_prep()
