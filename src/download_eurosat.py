from pathlib import Path
from collections import Counter
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ds = datasets.EuroSAT(root=str(DATA_DIR), download=True, transform=transforms.ToTensor())
labels = [ds[i][1] for i in range(len(ds))]
counts = Counter(labels)
idx_to_class = {i:c for i,c in enumerate(ds.classes)}

print("✅ EuroSAT downloaded.")
print("Images:", len(ds))
print("Classes:", ds.classes)
print("Per-class counts:", {idx_to_class[k]: v for k,v in sorted(counts.items())})
