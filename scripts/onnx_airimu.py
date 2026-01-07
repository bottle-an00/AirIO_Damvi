#!/usr/bin/env python3
import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn

# ============ 사용자 환경에 맞게 수정 ============
PROJECT_ROOT = "/home/jba/AirIO_Damvi"
CKPT_PATH    = "/home/jba/AirIO_Damvi/model/airimu/best_model.ckpt"

# ✅ 너가 YAML로 변환해둔 conf를 쓰는 걸 권장함
# 예: /home/jba/AirIO_Damvi/model/airimu/codenet.yaml
CONF_PATH    = "/home/jba/AirIO_Damvi/model/airimu/codenet.yaml"

ONNX_PATH    = "/home/jba/AirIO_Damvi/model/airimu/codenet_corr_feat.onnx"

B = 1
T = 50
# ==============================================

# (선택) legacy exporter + 일부 BLAS 백엔드에서 크래시(FPE) 방지용
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False

# 프로젝트 루트가 import 최우선이 되도록
sys.path.insert(0, PROJECT_ROOT)

def strip_prefix(state_dict, prefixes=("module.", "net.", "model.", "network.")):
    out = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out

def load_conf_yaml(path: str):
    from omegaconf import OmegaConf
    conf = OmegaConf.load(path)
    return conf

def build_codenet(conf):
    # CodeNet은 model/airimu/code.py 쪽에 있음
    from model.airimu.code import CodeNet
    # CodeNet은 conf.train을 받는 구조를 가정(기존 코드 패턴과 동일)
    codenet = CodeNet(conf.train)
    return codenet

class CodeNetCorrectionFeatWrapper(nn.Module):
    """
    Input : feat [B, 50, 6]
    Output: corr [B, 41, 6]  (= T-9)
    """
    def __init__(self, codenet: nn.Module):
        super().__init__()
        self.net = codenet

        # ✅ T=50 고정 export용 index map (len=41)
        idx = [0]*5 + [1]*9 + [2]*9 + [3]*9 + [4]*9  # 5 + 36 = 41
        self.register_buffer("idx_map", torch.tensor(idx, dtype=torch.long))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, 50, 6]
        acc  = feat[..., 0:3]
        gyro = feat[..., 3:6]

        # ✅ net.inference()를 쓰지 말고 encoder/decoder만 사용 (index_put 회피)
        x = torch.cat([acc, gyro], dim=-1)          # [B, 50, 6]
        h = self.net.encoder(x)[:, 1:, :]           # [B, 49, H]
        corr_feat = self.net.decoder(h)             # [B, 49, 6]

        corr_acc_feat  = corr_feat[..., 0:3]        # [B, 49, 3]
        corr_gyro_feat = corr_feat[..., 3:6]        # [B, 49, 3]

        # ✅ _update()와 동일한 결과를 gather로 생성: [B, 41, 3]
        corr_acc  = torch.index_select(corr_acc_feat,  dim=1, index=self.idx_map)
        corr_gyro = torch.index_select(corr_gyro_feat, dim=1, index=self.idx_map)

        corr = torch.cat([corr_acc, corr_gyro], dim=-1)  # [B, 41, 6]
        return corr

def main():
    # 1) conf 로드
    conf = load_conf_yaml(CONF_PATH)

    # (선택) export에서는 gtrot/propcov 같은 분기를 줄이는 게 안정적일 때가 많음
    # 네가 이미 gtrot False로 테스트했다고 했으니, 여기서도 덮어씀
    if hasattr(conf, "train"):
        conf.train.gtrot = False

    # 2) 모델 생성
    codenet = build_codenet(conf)
    codenet = codenet.float().eval()

    # 3) ckpt 로드
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = strip_prefix(ckpt["model_state_dict"])
    else:
        # 혹시 state_dict만 저장된 케이스
        state = strip_prefix(ckpt)

    missing, unexpected = codenet.load_state_dict(state, strict=False)
    print("missing keys:", len(missing), "unexpected keys:", len(unexpected))

    # 4) wrapper
    wrapper = CodeNetCorrectionFeatWrapper(codenet).eval()

    # 5) forward sanity
    dummy = (torch.randn(B, T, 6, dtype=torch.float32) * 0.01).clamp(-0.1, 0.1)
    with torch.no_grad():
        y = wrapper(dummy)
    print("forward ok:", y.shape, bool(torch.isfinite(y).all()))

    # 6) ONNX export (legacy)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy, strict=False)
        traced = torch.jit.freeze(traced)

    torch.onnx.export(
        wrapper,
        dummy,
        ONNX_PATH,
        input_names=["feat"],
        output_names=["corr"],
        opset_version=17,
        do_constant_folding=False,
        dynamic_axes=None
    )
    print("[OK] Exported:", ONNX_PATH)

    # 7) 간단 검증(onnxruntime)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"feat": dummy.numpy()})[0]
        print("ORT output shape:", out.shape, "expected:", (B, T - 9, 6))
    except Exception as e:
        print("[WARN] onnxruntime check skipped/failed:", repr(e))

if __name__ == "__main__":
    main()
