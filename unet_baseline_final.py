import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter

EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4
IMG_SIZE = 128
BASE = 32
DEBUG = False

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class PetDataset(Dataset):
    def __init__(self, split="trainval", root="./data", img_size=128, is_train=True):
        self.data = datasets.OxfordIIITPet(root=root, split=split,
                                           target_types="segmentation", download=True)
        self.size = img_size
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, mask = self.data[idx]
        img = img.convert("RGB")

        img = TF.resize(img, [self.size, self.size])
        mask = TF.resize(mask, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train and random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask_np = np.array(mask)
        mask_bin = (mask_np == 1).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0)

        return img, mask_tensor


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


class LossGop(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        loss1 = self.bce(pred, target)

        p = torch.sigmoid(pred).flatten(1)
        t = target.flatten(1)
        inter = (p * t).sum(1)
        dice = (2 * inter + 1e-6) / (p.sum(1) + t.sum(1) + 1e-6)
        loss2 = 1 - dice.mean()

        return loss1 + loss2


def danh_gia(model, loader, criterion, device):
    model.eval()
    tong_loss, tong_iou, tong_dice, so_anh = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            b = imgs.size(0)
            tong_loss += criterion(out, masks).item() * b

            pred = (torch.sigmoid(out) > 0.5).float().flatten(1)
            t = masks.flatten(1)
            inter = (pred * t).sum(1)
            union = pred.sum(1) + t.sum(1) - inter

            iou = (inter + 1e-6) / (union + 1e-6)
            dice = (2 * inter + 1e-6) / (pred.sum(1) + t.sum(1) + 1e-6)
            tong_iou += iou.sum().item()
            tong_dice += dice.sum().item()
            so_anh += b

    return tong_loss / so_anh, tong_iou / so_anh, tong_dice / so_anh


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("May dang dung:", device)

    print("Dang load anh...")
    full_train = PetDataset("trainval", img_size=IMG_SIZE, is_train=True)
    full_val = PetDataset("trainval", img_size=IMG_SIZE, is_train=False)
    test_ds = PetDataset("test", img_size=IMG_SIZE, is_train=False)

    n = len(full_train)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    n_val = int(n * 0.1)
    train_ds = Subset(full_train, idx[n_val:])
    val_ds = Subset(full_val, idx[:n_val])

    if DEBUG:
        train_ds = Subset(train_ds, range(100))
        val_ds = Subset(val_ds, range(30))
        test_ds = Subset(test_ds, range(30))

    print(f"So anh: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(in_c=3, out_c=1, base=BASE).to(device)
    so_tham_so = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"So tham so cua mo hinh: {so_tham_so:.2f} trieu")

    criterion = LossGop()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    writer = SummaryWriter('runs/unet_train_log')

    best_iou = 0.0
    best_epoch = 0
    print("Bat dau train...")
    t_bat_dau = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)

        val_loss, val_iou, val_dice = danh_gia(model, val_loader, criterion, device)

        writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
        writer.add_scalar('IoU/Val', val_iou, epoch)
        writer.add_scalar('Dice/Val', val_dice, epoch)

        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            torch.save(model.state_dict(), "unet_baseline.pth")
            print(" -> Da luu file unet_baseline.pth moi nhat")

    thoi_gian_train = (time.time() - t_bat_dau) / 60
    writer.close()
    print("Train hoan tat!")

    model.load_state_dict(torch.load("unet_baseline.pth", map_location=device))
    test_loss, test_iou, test_dice = danh_gia(model, test_loader, criterion, device)

    model.eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
    ms_moi_anh = (time.time() - t0) / 50 * 1000

    ket_qua = [
        ("Thiet bi", device),
        ("Kich thuoc anh", f"{IMG_SIZE}x{IMG_SIZE}"),
        ("Base channels", BASE),
        ("So tham so (trieu)", f"{so_tham_so:.2f}"),
        ("So epoch", EPOCHS),
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
    print("Da luu bang so lieu vao ket_qua_baseline.txt")