import torch
from tqdm import tqdm

class ACC:
    def __init__(self, model, model_defense=None):
        self.model = model
        self.model_defense = model_defense

    def evaluate(self, clean_loader, adv_loader):
        count = 0
        total = 0
        for (x, y) in tqdm(clean_loader):
            with torch.no_grad():
                x, y = x.cuda(), y.cuda()
                pred = self.model(x)
                pred = torch.argmax(pred, 1)
                total += x.shape[0]
                count += (pred == y).sum().item()
        acc = count / total
        return acc
                