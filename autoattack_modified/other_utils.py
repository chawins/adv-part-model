import copy
import os


class Logger:
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, "a") as f:
                f.write(str_to_log + "\n")
                f.flush()


def check_imgs(adv, x, norm):
    delta = (adv - x).view(adv.shape[0], -1)
    if norm == "Linf":
        res = delta.abs().max(dim=1)[0]
    elif norm == "L2":
        res = (delta**2).sum(dim=1).sqrt()
    elif norm == "L1":
        res = delta.abs().sum(dim=1)

    str_det = "max {} pert: {:.5f}, nan in imgs: {}, max in imgs: {:.5f}, min in imgs: {:.5f}".format(
        norm, res.max(), (adv != adv).sum(), adv.max(), adv.min()
    )
    print(str_det)

    return str_det


def L1_norm(x, keepdim=False):
    z = x.abs().view(x.shape[0], -1).sum(-1)
    if keepdim:
        z = z.view(-1, *[1] * (len(x.shape) - 1))
    return z


def L2_norm(x, keepdim=False):
    z = (x**2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1] * (len(x.shape) - 1))
    return z


def L0_norm(x):
    return (x != 0.0).view(x.shape[0], -1).sum(-1)


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_pred(output):
    if output.size(-1) == 1:
        return (output >= 0).squeeze().float()
    return output.argmax(1)


def mask_kwargs(kwargs_orig, batch_datapoint_idcs):
    kwargs = {}
    if "dino_targets" in kwargs_orig:
        kwargs = copy.deepcopy(kwargs_orig)
        kwargs["masks"] = kwargs_orig["masks"][batch_datapoint_idcs]
        kwargs["dino_targets"] = []

        if batch_datapoint_idcs.ndim == 0:
            kwargs["dino_targets"].append(
                kwargs_orig["dino_targets"][batch_datapoint_idcs]
            )
        else:
            for i in batch_datapoint_idcs:
                kwargs["dino_targets"].append(kwargs_orig["dino_targets"][i])
    return kwargs
