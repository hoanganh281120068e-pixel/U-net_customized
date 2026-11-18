import time
import torch
from unet_baseline_final import UNet

model = UNet(3, 1, 32)
ckpt = torch.load("unet_baseline.pth", map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()

x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    for _ in range(5):
        model(x)
    t0 = time.time()
    for _ in range(30):
        model(x)
ms = (time.time() - t0) / 30 * 1000
print(f"CPU: {ms:.2f} ms/anh, khoang {1000/ms:.1f} FPS")
