import os
import torch
from torch import nn
from torch.utils.data import  DataLoader
from torch import optim

from core.utils.dataset import NPYDataset
from core.utils.logger import pdebug, pinfo, pwarn
from core.utils.yaml_parser import parser
from core.evaluate import *
from model.module import *

class EvaluateCore:
    """
    评测指标基类，评测过程在此类中进行
    """
    def __init__(self, model, args):
        self.args = args["evaluate"]
        self.model = model

        # 读取模型参数
        pdebug("Loading model parameters...")
        if "model_path" in self.args:
            model_path = self.args["model_path"]
            self.model.load_state_dict(torch.load(model_path))
        else:
            pdebug("Using model from defense or attack.")
        
        # 读取对比模型
        self.model_defense = None
        if "model_defense" in self.args:
            pdebug("Loading compared model parameters...")
            self.model_defense = eval(self.args["model_defense"])()
            self.model_defense.load_state_dict(torch.load(self.args["model_defense_path"]))

        # 读取评测数据
        pdebug("Getting evaluate clean data...")
        data_path = self.args["clean"]["data_path"]
        label_path = self.args["clean"]["label_path"]
        batch_size = self.args["clean"]["batch_size"]
        self.clean_loader = DataLoader(NPYDataset(data_path, label_path), batch_size=batch_size)

        self.adv_loader = None
        if "adv" in self.args:
            pdebug("Getting evaluate adv data...")
            data_path = self.args["adv"]["data_path"]
            label_path = self.args["adv"]["label_path"]
            batch_size = self.args["adv"]["batch_size"]
            self.adv_loader = DataLoader(NPYDataset(data_path, label_path), batch_size=batch_size)

        # 创建评测基类
        pdebug("Creating evaluate base...")
        self.attack_class = eval(self.args["method"])(self.model, self.model_defense)

    def process(self):
        # 开始评测
        result = self.attack_class.evaluate(self.clean_loader, self.adv_loader)
        print(self.args["method"], result)