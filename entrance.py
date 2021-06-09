import sys

from core.utils.yaml_parser import arg_parser
from core import AttackCore, EvaluateCore
import importlib

if __name__ == "__main__":
    device = "cpu"
    args = arg_parser(sys.argv[1])

    # 获取模型
    module_user = importlib.import_module(args["model"])
    model = module_user.getModel()
    model = model.load_state_dict(torch.load(model_path))
    model = model.eval()
    model.to(device)

    if "attack" in args:
        attack_core = AttackCore(model, args)
        attack_core.process()

    if "evaluate" in args:
        evaluate_core = EvaluateCore(model, args)
        evaluate_core.process()