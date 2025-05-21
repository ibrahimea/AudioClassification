# train_binary_full_fixed.py — Enhanced with Label Smoothing & OneCycleLR Scheduler (Fixed Indentation & Duplicates)
import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# -------- Configurable Model Variant --------
MODEL_VARIANT = 'resnet50'  # options: 'resnet50', 'resnet101'
NUM_CLASSES   = 2
DROPOUT_P      = 0.5

# -------- Indoor vs Outdoor Labels --------
indoor = {
    'toilet_flush'
}
outdoor = {
    'car_horn'
}

# -------- Create Model --------
def create_model(variant='resnet50'):
    if variant == 'resnet50':
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif variant == 'resnet101':
        backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown variant {variant}")
    # freeze all except layer3, layer4 & fc
    for name, p in backbone.named_parameters():
        if not (name.startswith('layer3') or name.startswith('layer4') or name.startswith('fc')):
            p.requires_grad = False
    # replace head
    in_fc = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Linear(in_fc, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(DROPOUT_P),
        nn.Linear(256, NUM_CLASSES)
    )
    return backbone

# -------- Dataset --------
class AudioSpectrogramDataset(Dataset):
    def __init__(self, root_dir, T_max, transform=None,
                 sample_rate=44100, n_fft=1024):
        self.T_max = T_max
        self.transform = transform
        self.sr = sample_rate
        self.n_fft = n_fft
        self.files = []
        self.labels = []
        for cls in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, cls)
            if cls  not in indoor and cls not in outdoor:
                continue
            if not os.path.isdir(class_dir):
                continue
            label = 0 if cls in indoor else 1
            for root_subdir, _, filenames in os.walk(class_dir):
                for fname in filenames:
                    if fname.lower().endswith(('.wav', '.flac', '.mp3')):
                        self.files.append(os.path.join(root_subdir, fname))
                        self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def _pad_truncate(self, spec):
        T = spec.shape[1]
        if T < self.T_max:
            return F.pad(spec, (0, self.T_max - T, 0, 0))
        return spec[:, :self.T_max]

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        # load audio
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        if len(y) < self.n_fft:
            y = np.pad(y, (0, self.n_fft - len(y)), mode='constant')
        # compute Mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=self.n_fft, n_mels=128
        )
        spec_db = librosa.power_to_db(mel, ref=np.max)
        spec = torch.from_numpy(spec_db).float()
        # pad/truncate
        spec = self._pad_truncate(spec)
        # convert to 3-channel image
        img = spec.unsqueeze(0).repeat(3, 1, 1)
        if self.transform:
            img = self.transform(img)
        return img, label

# -------- MixUp --------
def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

# -------- Main --------
# L1 & L2 regularization coefficients\ nLR1 = 1e-5  # L1 regularization strength
LR2 = 1e-4  # L2 regularization strength
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    # settings
    ROOT = 'Data'
    T_MAX = 400
    BATCH = 32
    EPOCHS = 100
    CKPT = 'binary_checkpoint.pth'
    PATIENCE = 45
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    torch.manual_seed(42)

    # dataset & splits
    ds = AudioSpectrogramDataset(ROOT, T_MAX, transform)
    n = len(ds)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n - n_train - n_val])

    # weighted sampler for train
    train_labels = [ds.labels[i] for i in train_ds.indices]
    counts = np.bincount(train_labels, minlength=2)
    class_weights = 1.0 / counts
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=4)

    # model, optimizer, scheduler, loss
    model = create_model(MODEL_VARIANT).to(DEVICE)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # resume
    start_epoch = 1
    best_val = float('inf')
    wait = 0
    if args.resume and os.path.exists(CKPT):
        ckpt = torch.load(CKPT, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt.get('scheduler_state_dict', {}))
        start_epoch = ckpt['epoch'] + 1
        best_val = ckpt['best_val_loss']
        print(f"Resumed from epoch {start_epoch}")

    # training loop
    for epoch in range(start_epoch, EPOCHS + 1):
        # train
        model.train()
        train_loss = 0.0
        train_correct = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()
        train_loss /= n_train
        train_acc = train_correct / n_train

        # validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()
        val_loss /= n_val
        val_acc = val_correct / n_val

        print(f"Epoch {epoch}/{EPOCHS}  Train: loss={train_loss:.4f}, acc={train_acc:.4f}  "
              f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}")

        # checkpoint & early stop
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            wait = 0
        else:
            wait += 1
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val
        }
        torch.save(ckpt, CKPT)
        if is_best:
            print("  → New best saved")
        if wait > PATIENCE:
            print("Early stopping triggered")
            break

    # final test
    final_ckpt = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(final_ckpt['model_state_dict'])
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            test_correct += (out.argmax(1) == labels).sum().item()
    print(f"Test Acc: {test_correct/len(test_ds):.4f}")
