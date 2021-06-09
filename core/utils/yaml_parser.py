import os
import yaml


def parser(path):
    with open(path, "r") as fp:
        args = yaml.load(fp, Loader=yaml.FullLoader)
    return args

def arg_parser(dir_path):
    args = parser(os.path.join(dir_path, "config.yaml"))
        
    # 填充默认值
    args["config_path"] = dir_path
    if "attack" in args:
        if "clean" in args["attack"]:
            if "batch_size" not in args["attack"]["clean"]:
                args["attack"]["clean"]["batch_size"] = 64

    if "evaluate" in args:
        if "clean" in args["evaluate"]:
            if "batch_size" not in args["evaluate"]["clean"]:
                args["evaluate"]["clean"]["batch_size"] = 64
        if "adv" in args["evaluate"]:
            if "batch_size" not in args["evaluate"]["adv"]:
                args["evaluate"]["adv"]["batch_size"] = 64

    return args


if __name__ == "__main__":
    arg_parser("settings/default")