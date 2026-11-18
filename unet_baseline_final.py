import sys
from pathlib import Path
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

DUONG_DAN_DATA = Path(r"D:\hoc_it\TTNT\TTNT")
if not DUONG_DAN_DATA.exists():
    DUONG_DAN_DATA = Path(__file__).resolve().parent
sys.path.insert(0, str(DUONG_DAN_DATA))

import data_pipeline as dp

EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4
BASE = 32
IMG_SIZE = 256
DEBUG = False

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_c=3, out_c=1, base=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.e1 = DoubleConv(in_c, base)
        self.e2 = DoubleConv(base, base * 2)
        self.e3 = DoubleConv(base * 2, base * 4)
        self.e4 = DoubleConv(base * 4, base * 8)

        self.bot = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.d4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_c, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3))
        b = self.bot(self.pool(x4))

        x = self.up4(b)
        x = self.d4(torch.cat([x4, x], dim=1))

        x = self.up3(x)
        x = self.d3(torch.cat([x3, x], dim=1))

        x = self.up2(x)
        x = self.d2(torch.cat([x2, x], dim=1))

        x = self.up1(x)
        x = self.d1(torch.cat([x1, x], dim=1))

        return self.out(x)


def lay_batch(tap, batch_size, debug=False):
    so_batch = 0
    for anh, mask in dp.tao_batch_unet(tap=tap, batch_size=batch_size):
        x = torch.from_numpy(anh)
        m = torch.from_numpy(mask).float()
        hop_le = (m != 255).float().unsqueeze(1)
        nhan = (m == 1).float().unsqueeze(1)
        yield x, nhan, hop_le
        so_batch += 1
        if debug and so_batch >= 3:
            break


def tinh_loss(logits, nhan, hop_le, eps=1e-6):
    bce_tung_pixel = nn.functional.binary_cross_entropy_with_logits(
        logits, nhan, reduction="none")
    bce = (bce_tung_pixel * hop_le).sum() / (hop_le.sum() + eps)

    p = torch.sigmoid(logits) * hop_le
    t = nhan * hop_le
    inter = (p * t).sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + eps)

    return bce + (1 - dice.mean())


@torch.no_grad()
def iou_dice_tung_anh(logits, nhan, hop_le, eps=1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float() * hop_le
    t = nhan * hop_le
    inter = (pred * t).sum(dim=(1, 2, 3))
    tong_pred = pred.sum(dim=(1, 2, 3))
    tong_nhan = t.sum(dim=(1, 2, 3))
    iou = (inter + eps) / (tong_pred + tong_nhan - inter + eps)
    dice = (2 * inter + eps) / (tong_pred + tong_nhan + eps)
    return iou, dice


@torch.no_grad()
def danh_gia(model, tap, device):
    model.eval()
    tong_loss, tong_iou, tong_dice, so_anh = 0.0, 0.0, 0.0, 0
    for x, nhan, hop_le in lay_batch(tap, BATCH_SIZE, DEBUG):
        x, nhan, hop_le = x.to(device), nhan.to(device), hop_le.to(device)
        out = model(x)
        b = x.size(0)
        tong_loss += tinh_loss(out, nhan, hop_le).item() * b
        iou, dice = iou_dice_tung_anh(out, nhan, hop_le)
        tong_iou += iou.sum().item()
        tong_dice += dice.sum().item()
        so_anh += b
    return tong_loss / so_anh, tong_iou / so_anh, tong_dice / so_anh, so_anh


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("May dang dung:", device)
    print("So mau: train=%d, val=%d, test=%d" % (
        len(dp.CAC_TAP["train"]), len(dp.CAC_TAP["val"]), len(dp.CAC_TAP["test"])))

    model = UNet(in_c=3, out_c=1, base=BASE).to(device)
    so_tham_so = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"So tham so: {so_tham_so:.2f} trieu")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    writer = SummaryWriter('runs/unet_train_log')

    lich_su_train_loss, lich_su_val_loss = [], []
    lich_su_val_iou, lich_su_val_dice = [], []

    best_iou = 0.0
    best_epoch = 0
    print("Bat dau train...")
    t_bat_dau = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tong_loss, so_batch = 0.0, 0
        for x, nhan, hop_le in lay_batch("train", BATCH_SIZE, DEBUG):
            x, nhan, hop_le = x.to(device), nhan.to(device), hop_le.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = tinh_loss(out, nhan, hop_le)
            loss.backward()
            optimizer.step()

            tong_loss += loss.item()
            so_batch += 1
        train_loss = tong_loss / so_batch

        val_loss, val_iou, val_dice, _ = danh_gia(model, "val", device)

        lich_su_train_loss.append(train_loss)
        lich_su_val_loss.append(val_loss)
        lich_su_val_iou.append(val_iou)
        lich_su_val_dice.append(val_dice)

        writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
        writer.add_scalar('IoU/Val', val_iou, epoch)
        writer.add_scalar('Dice/Val', val_dice, epoch)
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(),
                        "config": {"in_c": 3, "out_c": 1, "base": BASE, "img_size": IMG_SIZE}},
                       "unet_baseline.pth")
            print(" -> Da luu unet_baseline.pth")

    thoi_gian_train = (time.time() - t_bat_dau) / 60
    writer.close()
    print("Train hoan tat!")

    ckpt = torch.load("unet_baseline.pth", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_loss, test_iou, test_dice, so_test = danh_gia(model, "test", device)
    if not DEBUG:
        assert so_test == len(dp.CAC_TAP["test"]), "Thieu mau trong tap test"

    model.eval()
    x_dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        for _ in range(5):
            model(x_dummy)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            model(x_dummy)
        if device == 'cuda':
            torch.cuda.synchronize()
    ms_moi_anh = (time.time() - t0) / 50 * 1000

    ket_qua = [
        ("Thiet bi", device),
        ("Tap du lieu", "splits cua Thanh vien 1"),
        ("So mau train / val / test", f"{len(dp.CAC_TAP['train'])} / {len(dp.CAC_TAP['val'])} / {len(dp.CAC_TAP['test'])}"),
        ("Kich thuoc anh", f"{IMG_SIZE}x{IMG_SIZE}"),
        ("Chuan hoa dau vao", "[0, 1]"),
        ("Pixel vien (255)", "bo qua khi tinh loss va metric"),
        ("Base channels", BASE),
        ("So tham so (trieu)", f"{so_tham_so:.2f}"),
        ("So epoch", EPOCHS),
        ("Batch size / learning rate", f"{BATCH_SIZE} / {LR}"),
        ("Epoch tot nhat", best_epoch),
        ("Thoi gian train (phut)", f"{thoi_gian_train:.2f}"),
        ("Val IoU tot nhat", f"{best_iou:.4f}"),
        ("Test IoU", f"{test_iou:.4f}"),
        ("Test Dice", f"{test_dice:.4f}"),
        ("Toc do du doan (ms/anh)", f"{ms_moi_anh:.2f}"),
    ]
    print("\n===== BANG SO LIEU BASELINE =====")
    with open("ket_qua_baseline.txt", "w", encoding="utf-8") as f:
        for ten, gia_tri in ket_qua:
            dong = f"{ten}: {gia_tri}"
            print(dong)
            f.write(dong + "\n")
    print("Da luu ket_qua_baseline.txt")

    epochs_range = range(1, len(lich_su_train_loss) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, lich_su_train_loss, label='Train Loss', marker='o')
    plt.plot(epochs_range, lich_su_val_loss, label='Val Loss', marker='o')
    plt.title('Bieu do Loss qua cac Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, lich_su_val_iou, label='Val IoU', marker='s')
    plt.plot(epochs_range, lich_su_val_dice, label='Val Dice', marker='^')
    plt.title('Bieu do IoU va Dice tren tap Val')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("bieu_do_ket_qua.png", dpi=150)
    plt.close()
    print("Da luu bieu_do_ket_qua.png")

    x_mau, nhan_mau, hop_le_mau = next(lay_batch("test", 4, debug=True))
    with torch.no_grad():
        pred_mau = (torch.sigmoid(model(x_mau.to(device))) > 0.5).float().cpu()

    so_hinh = min(4, x_mau.size(0))
    fig, axes = plt.subplots(so_hinh, 3, figsize=(9, 3 * so_hinh))
    for i in range(so_hinh):
        img_np = x_mau[i].permute(1, 2, 0).numpy()
        gt_np = nhan_mau[i, 0].numpy()
        pr_np = pred_mau[i, 0].numpy()

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Anh goc")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_np, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pr_np, cmap="gray")
        axes[i, 2].set_title("U-Net Du doan")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("du_doan_minh_hoa.png", dpi=150)
    plt.close()
    print("Da luu du_doan_minh_hoa.png")