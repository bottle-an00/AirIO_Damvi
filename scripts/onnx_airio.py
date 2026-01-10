#!/usr/bin/env python3

import torch, sys
from collections import OrderedDict
from pyhocon import ConfigFactory

PROJECT_ROOT = "/root/AirIO_Damvi"
sys.path.insert(0, PROJECT_ROOT)

from model.airio import net_dict


# ====== DEFAULT SETTINGS (사용자 확정 사항) ======
DEFAULT_CONFIG = "/root/AirIO_Damvi/model/airio/motion_body_rot.conf"
DEFAULT_CKPT = "/root/AirIO_Damvi/model/airio/best_model.ckpt"
DEFAULT_OUT = "/root/AirIO_Damvi/model/airio/airio_codewithrot_fp32_T50.onnx"

BATCH_SIZE = 1
SEQ_LEN = 1000
OPSET_VERSION = 17
DTYPE = torch.float32
# ================================================


def _clean_state_dict(sd):
    if "state_dict" in sd:
        sd = sd["state_dict"]

    out = OrderedDict()
    for k, v in sd.items():
        for pref in ("model.", "network.", "net.", "module."):
            if k.startswith(pref):
                k = k[len(pref):]
        out[k] = v
    return out


class OnnxWrapper(torch.nn.Module):
    """
    dict 출력 → (cov, net_vel) 튜플 출력으로 변환
    """
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, acc, gyro, rot):
        data = {
            "acc": acc,
            "gyro": gyro,
        }
        out = self.net.forward(data, rot=rot)
        return out["cov"], out["net_vel"]


def main():
    # 1) config
    conf = ConfigFactory.parse_file(DEFAULT_CONFIG)

    # 2) network (codewithrot 고정)
    net = net_dict["codewithrot"](conf.train)
    net.eval()
    net.float()

    # 3) checkpoint
    ckpt = torch.load(DEFAULT_CKPT, map_location="cpu")
    sd = _clean_state_dict(ckpt)
    net.load_state_dict(sd, strict=False)

    # 4) wrapper
    wrapper = OnnxWrapper(net).eval().float()

    # 5) dummy inputs (FP32, T=50 고정)
    acc = torch.zeros((BATCH_SIZE, SEQ_LEN, 3), dtype=DTYPE)
    gyro = torch.zeros((BATCH_SIZE, SEQ_LEN, 3), dtype=DTYPE)
    rot = torch.zeros((BATCH_SIZE, SEQ_LEN, 3), dtype=DTYPE)

    # 6) export
    torch.onnx.export(
        wrapper,
        (acc, gyro, rot),
        DEFAULT_OUT,
        opset_version=OPSET_VERSION,
        input_names=["acc", "gyro", "rot"],
        output_names=["cov", "net_vel"],
        do_constant_folding=True,
    )

    print(f"[OK] ONNX exported: {DEFAULT_OUT}")


if __name__ == "__main__":
    main()
