import numpy as np

def parse_ckpts(original_ckpt_list):
    if len(original_ckpt_list) == 0:
        return original_ckpt_list
    ckpt_step_list = []
    i = 0
    while len(original_ckpt_list) > i and original_ckpt_list[i] == -1:
        if len(original_ckpt_list) < i+4:
            raise ValueError("the input ckpt-step-list hs wrong format")
        ckpt_step_list += list(range(original_ckpt_list[i+1], original_ckpt_list[i+2]+1, original_ckpt_list[i+3]))
        i += 4
    if len(original_ckpt_list) > i:
        ckpt_step_list += original_ckpt_list[i:]
    return ckpt_step_list
