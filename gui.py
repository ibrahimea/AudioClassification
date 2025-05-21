# gui_binary.py

import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from torchvision import transforms, models

BINARY_CLASSES = ['indoor','outdoor']

DATA_ROOT   = 'Data'
CHECKPOINT  = 'binary_checkpoint.pth'
T_MAX       = 400
SAMPLE_RATE = 44100
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features,256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256,2)
)
model = model.to(DEVICE)

if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(f"Checkpoint '{CHECKPOINT}' not found.")
# Load the checkpoint (a dict), then extract and load only the model weights
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def predict_audio(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(y) < 1024:
        y = np.pad(y, (0, 1024-len(y)), mode='constant')
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=1024, n_mels=128)
    spec_db = librosa.power_to_db(mel, ref=np.max)
    spec = torch.from_numpy(spec_db).float()
    T = spec.shape[1]
    if T< T_MAX:
        spec = F.pad(spec,(0,T_MAX-T,0,0))
    else:
        spec = spec[:,:T_MAX]
    img = spec.unsqueeze(0).repeat(3,1,1)
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img)
        cls = out.argmax(1).item()
    return BINARY_CLASSES[cls]

class AudioClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Indoor vs Outdoor Audio")
        self.geometry("400x200")
        tk.Label(self, text="Select an audio file to classify").pack(pady=20)
        tk.Button(self, text="Browse", command=self.run).pack()

    def run(self):
        path = filedialog.askopenfilename(
            filetypes=[("Audio","*.wav *.flac *.mp3"),("All","*")]
        )
        if not path: return
        try:
            cls = predict_audio(path)
            messagebox.showinfo("Result", f"This is: {cls.upper()}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    AudioClassifierApp().mainloop()
