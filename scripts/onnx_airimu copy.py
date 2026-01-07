import sys
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from omegaconf import OmegaConf
from collections import OrderedDict

# 🔧 경로 설정
sys.path.insert(0, "/home/jba/AirIO_Damvi")

CKPT_PATH = "/home/jba/AirIO_Damvi/model/airimu/best_model.ckpt"
CONF_PATH = "/home/jba/AirIO_Damvi/model/airimu/codenet.yaml"   # 네가 업로드한 conf
ONNX_PATH = "codenet_corr.onnx"

# ONNX export용 window size (고정 권장)
T = 50
B = 1

# ------------------------
# state_dict helper
# ------------------------
def strip_prefix(state_dict, prefixes=("module.", "net.", "model.", "network.")):
    out = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out

# ------------------------
# Wrapper (feat → correction)
# ------------------------
class CodeNet2InputWrapper(nn.Module):
    def __init__(self, codenet):
        super().__init__()
        self.net = codenet

    def forward(self, acc: torch.Tensor, gyro: torch.Tensor) -> torch.Tensor:
        # acc : [B, T, 3]
        # gyro: [B, T, 3]
        out = self.net.inference({"acc": acc, "gyro": gyro})
        corr = torch.cat([out["correction_acc"], out["correction_gyro"]], dim=-1)  # [B, T-9, 6]
        return corr

# ------------------------
# main
# ------------------------
def main():
    # (1) conf 로드
    conf = OmegaConf.load(CONF_PATH)

    # (2) codenet 생성
    from model.airimu.code import CodeNet
    codenet = CodeNet(conf.train)

    # (3) checkpoint 로드
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = strip_prefix(ckpt["model_state_dict"])
    codenet.load_state_dict(state_dict, strict=False)

    # (4) dtype / eval
    codenet = codenet.float().eval()

    # (5) ONNX export
    dummy_acc  = torch.randn(B, T, 3, dtype=torch.float32) * 0.01
    dummy_gyro = torch.randn(B, T, 3, dtype=torch.float32) * 0.01

    wrapper = CodeNet2InputWrapper(codenet).eval()

    torch.onnx.export(
        wrapper,
        (dummy_acc, dummy_gyro),
        "codenet_corr.onnx",
        input_names=["acc", "gyro"],
        output_names=["corr"],
        opset_version=17,
        dynamo=True,
    )

    print("[OK] Exported:", ONNX_PATH)

    # (7) ONNX check
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)

    # (8) ONNXRuntime test
    sess = ort.InferenceSession("codenet_corr.onnx", providers=["CPUExecutionProvider"])
    print([(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print([(o.name, o.shape, o.type) for o in sess.get_outputs()])

if __name__ == "__main__":
    main()
