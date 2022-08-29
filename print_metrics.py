import json
import os

import numpy as np

GEN_GAP_EVAL_EPOCHS = 20
exp_path = "./results/"
exp_list = os.listdir(exp_path)
exp_list = [os.path.join(exp_path, exp, "log.txt") for exp in exp_list]
exp_list = [exp for exp in exp_list if os.path.isfile(exp)]
exp_list = sorted(exp_list)

for exp in exp_list:
    with open(exp) as file:
        lines = [line.rstrip() for line in file]
    data = [json.loads(line) for line in lines if "test_acc1" in line]
    if len(data) == 0:
        continue
    train_acc = np.array([d["train_acc1"] for d in data][-GEN_GAP_EVAL_EPOCHS:])
    val_acc = np.array([d["test_acc1"] for d in data][-GEN_GAP_EVAL_EPOCHS:])
    if not (len(train_acc) == len(val_acc) == GEN_GAP_EVAL_EPOCHS):
        continue
    avg_gen_gap = np.mean((train_acc - val_acc) / train_acc)
    print(f'{exp.split("/")[2]}: avg_gen_gap = {avg_gen_gap}')
