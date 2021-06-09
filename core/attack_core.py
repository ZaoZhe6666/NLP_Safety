import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from core.utils.dataset import NPYDataset
from core.utils.logger import pdebug, pinfo, pwarn
from core.utils.yaml_parser import parser
from core.attack import *


class AttackCore:
    """
    攻击算法基类，攻击过程在此类中进行
    """

    def __init__(self, model, args):
        self.args = args["attack"]
        self.model = model

        # 读取攻击样本
        pdebug("Getting attack clean data...")
        data_path = self.args["clean"]["data_path"]
        label_path = self.args["clean"]["label_path"]
        batch_size = self.args["clean"]["batch_size"]
        self.clean_loader = DataLoader(NPYDataset(data_path, label_path), batch_size=batch_size)

        # 创建攻击基类
        pdebug("Creating attack base...")
        attack_args = parser(self.args["config_path"])
        self.attack_class = eval(self.args["method"])(self.model, attack_args)

    def process(self):
        # 开始攻击
        self.model.eval()
        adv_x = []
        adv_y = []

        num_classes = 0
        for x, y in tqdm(self.clean_loader):
            x, y = x.cuda(), y.cuda()
            adv = self.attack_class.attack(x, y)
            if "adv" in self.args:
                with torch.no_grad():
                    adv_label = self.model(adv).cpu().numpy()
                    num_classes = adv_label.shape[1]
                    adv_label = np.argmax(adv_label, 1)
                    adv_y.extend(list(adv_label))
            adv = adv.cpu().numpy()
            adv_x.extend(list(adv))

        adv_x = np.array(adv_x)
        adv_y = np.array(adv_y)
        adv_yt = np.zeros((adv_y.shape[0], num_classes))
        for i in range(adv_y.shape[0]):
            adv_yt[i][adv_y[i]] = 1

        # 保存对抗样本
        if "adv" in self.args:
            data_path = self.args["adv"]["data_path"]
            label_path = self.args["adv"]["label_path"]
            np.save(data_path, adv_x)
            np.save(label_path, adv_yt)
            


