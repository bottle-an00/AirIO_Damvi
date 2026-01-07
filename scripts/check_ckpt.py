import torch
ckpt_path = "/home/jba/AirIO_Damvi/model/airimu/best_model.ckpt"

obj = torch.load(ckpt_path, map_location="cpu")

print("type:", type(obj))
if isinstance(obj, dict):
    print("keys:", list(obj.keys())[:50])
    # 흔한 케이스들:
    for k in ["state_dict", "model", "net", "network", "ema", "module", "model_state_dict"]:
        if k in obj:
            v = obj[k]
            print(f"found key='{k}', type={type(v)}")
else:
    # state_dict 자체로 저장된 경우일 수 있음
    pass
