#!/usr/bin/env python
# -*- coding:utf-8 -*-
""" 
@author: romanshen 
@file: prep_data.py 
@time: 2021/05/06
@contact: xiangqing.shen@njust.edu.cn
"""


import os

import hydra
from omegaconf import OmegaConf
import pickle

from prep_utils import Lang, prepare_data, prepare_test_data


def run(cfg):
    os.chdir(hydra.utils.get_original_cwd())
    input_lang = Lang("input")
    pl1 = Lang("position")
    char = Lang("char")
    rule_lang = Lang("rule")
    raw_train = list()

    input_lang, pl1, char, rule_lang, raw_train = prepare_data(
        cfg.datadir, cfg.event, input_lang, pl1, char, rule_lang, raw_train
    )
    input_lang, pl1, char, rule_lang, raw_train = prepare_data(
        cfg.datadir2,
        cfg.event,
        input_lang,
        pl1,
        char,
        rule_lang,
        raw_train,
        cfg.rule_mappings,
    )

    input2_lang, pl2, char2, rule_lang2, raw_dev = prepare_data(
        cfg.dev_datadir, cfg.event, valids="rule_mappings.json"
    )

    input3_lang, pl3, char3, rule_lang3, raw_test1 = prepare_test_data(cfg.dev_datadir)
    input3_lang, pl3, char3, rule_lang3, raw_test2 = prepare_test_data(cfg.test_datadir)

    os.mkdir(cfg.outdir)

    with open("%s/train" % cfg.outdir, "wb") as f:
        pickle.dump((input_lang, pl1, char, rule_lang, raw_train), f)
    with open("%s/dev" % cfg.outdir, "wb") as f:
        pickle.dump((input2_lang, pl2, char2, rule_lang2, raw_dev), f)
    with open("%s/test1" % cfg.outdir, "wb") as f:
        pickle.dump((input3_lang, pl3, char3, rule_lang3, raw_test1), f)
    with open("%s/test2" % cfg.outdir, "wb") as f:
        pickle.dump((input3_lang, pl3, char3, rule_lang3, raw_test2), f)


@hydra.main(config_path="conf", config_name="conf_ge_data")
def run_prep(cfg):
    print(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    run_prep()
