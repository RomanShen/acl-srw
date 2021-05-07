#!/usr/bin/env python
# -*- coding:utf-8 -*-
""" 
@author: romanshen 
@file: run_srw.py 
@time: 2021/05/07
@contact: xiangqing.shen@njust.edu.cn
"""

import logging
import os
import datetime

from omegaconf import OmegaConf
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


logger = logging.getLogger(__name__)
