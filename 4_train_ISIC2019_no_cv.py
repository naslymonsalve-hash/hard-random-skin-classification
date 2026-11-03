# =============================================================================
# DCNN Model for ISIC2019 Skin Lesion Classification (PyTorch)
# SOTA Implementation with Hard-Sample Mining, Mixed Augmentation, TTA, and Timing
# =============================================================================

import os
import json
import hashlib
from pickletools import optimize
import random
import copy
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from pathlib import Path
from PIL import Image

import wandb
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GroupShuffleSplit


# =============================================================================
# 1. GLOBAL CONFIGURATIONS & HYPERPARAMETERS
# =============================================================================

# --- Environment & Reproducibility ---
SEED = int(os.getenv("EXPERIMENT_SEED", "42")) # 42, 123, 2024
# Seed for the only train/validation split; it is NOT the training seed.
SPLIT_SEED = int(os.getenv("EXPERIMENT_SPLIT_SEED", "42"))
USE_AMP = torch.cuda.is_available()  # Automatic Mixed Precision for faster training
USE_WANDB = False                           # <--- Activate/Deactivate WandB
WANDB_PROJECT = "ISIC2019_HARD_RANDOM_Holdout"   # <--- Project name on the panel

# --- Directory Paths ---
DATA_DIR = r"D:\Skin Cancer\ISIC2019\ISIC_2019_Training_Input"
DATA_TEST_DIR = r"D:\Skin Cancer\ISIC2019\ISIC_2019_Test_Input"
#OUTPUT_DIR = r"E:\Skin Cancer\Pipeline\results_efficientetnetb0_ISIC2019_3classes"
## 1. Load the ISIC metadata CSV (Adjust the path to where your CSV is saved)
metadata_path = r"D:\Skin Cancer\ISIC2019\ISIC_2019_Training_Metadata.csv"


# --- Dataset & Classes ---
#CLASS_NAMES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
CLASS_NAMES = ["BCC", "MEL", "SCC"] # 3 ISIC2019 classes
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

chosen_model = "NasNet"  # Keeps the MnasNet-100 backbone from the original experiment.
if chosen_model == "EfficientNet":
    from efficientnet_isic2019_model_block import EfficientNet_ISIC2019
EFFICIENTNET_VARIANT = "efficientnet_b0"

# --- Neural Network Hyperparameters ---
IMG_SIZE = (224,224) #(64, 64)       # Input size for the CNN
BATCH_SIZE = 32                      # Number of samples per batch
EPOCHS = 500                         # Maximum number of training epochs
LR = 6e-4                            # Initial Learning Rate (Adamax)
FOCAL_LOSS_GAMMA = 2.0               # Focusing parameter for Focal Loss (handles imbalance)

# --- Scheduler & Early Stopping ---
LR_PATIENCE = 12 #12                 # Epochs to wait before reducing LR on plateau
LR_FACTOR = 1/3 #0.5                 # Factor by which the learning rate will be reduced
MIN_LR = 1e-6                        # Lower bound on the learning rate
ES_PATIENCE = 15 #15                 # Epochs to wait for improvement before early stopping

# --- Advanced Trainin Dynamics (Hard Mining & Augmentation) ---
USE_RANDOM_OVERSAMPLING = False      # True to activate, False to deactivate.
#HERE I MAKE CHANGES -----------------------------------------------------------------------------------------------------------------------
DROP_RATE = float(os.getenv("EXPERIMENT_DROP_RATE", "0.4")) # 0 to 0.9
SELECTION_METHOD = os.getenv("EXPERIMENT_SELECTION_METHOD", "hard").lower() # hard | random
CV_FOLD = int(os.getenv("EXPERIMENT_CV_FOLD", "-1"))
if CV_FOLD != -1:
    raise ValueError("This copy is only for a fixed split without CV: use EXPERIMENT_CV_FOLD=-1.")
VALIDATION_FRACTION = float(os.getenv("EXPERIMENT_VALIDATION_FRACTION", "0.20"))
if not 0 < VALIDATION_FRACTION < 1:
    raise ValueError("The validation fraction must be between 0 and 1.")
if SELECTION_METHOD not in {"hard", "random"} or not 0 <= DROP_RATE < 1:
    raise ValueError("Invalid deletion method or rate.")
CV_N_SPLITS = 5  # Kept only for compatibility; not used in this copy.
CSV_OPTIONS = {"sep": ";", "decimal": ",", "index": False}
UPDATE_LOSS_FREQ = 5                 # Updates the loss of training samples every N epochs
AUGMENTATION_PROB = 0 #0.66          # Probability of applying CutMix or MixUp
HARD_MINING_TEMPERATURE = 1.0        # Temperature for stable softmax in sample re-selection
MIN_KEEP_PER_CLASS = 0               # Minimum number of samples guaranteed per class
COMPUTE_FULL_TRAIN_LOSS = False      # Set True to accurately calculate full train loss (computationally expensive)
WARMUP_EPOCHS = 0                    # Epochs where only the classifier (frozen backbone) is trained.
RAMP_WARMUP_EPOCHS = 20              # Number of smooth ramp-up periods of the learning rate

# Converts 0.0 → D0, 0.5 → D05 and 0.7 → D07
drop_rate_name = f"{DROP_RATE:.1f}".replace(".", "")
OUTPUT_ROOT = os.getenv("EXPERIMENT_OUTPUT_ROOT", str(Path(__file__).resolve().parent / "results_10seeds_no_cv"))
fold_name = "_NoCV"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f"results_{chosen_model}_ISIC2019_3classes_{SELECTION_METHOD}_r{drop_rate_name}_Seed{SEED}{fold_name}")

# --- Image Processing (CLAHE) ---
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# -----------------------------------------------------------------------------
# Derived Config Variables
# -----------------------------------------------------------------------------
label_to_index = {label: idx for idx, label in enumerate(CLASS_NAMES)}
index_to_label = {idx: label for label, idx in label_to_index.items()}


# =============================================================================
# 2. GENERAL UTILITIES & SETUP
# =============================================================================

def set_seed(seed=SEED):
    """Fixes all random seeds to ensure reproducible results across runs."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #Final version (for reproducibility of results)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        
        ##For testing (quick execution)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# Initialize environment
set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_dataframe(data_dir):
    """Scans the directory and builds a DataFrame mapping filepaths to labels."""
    data = []
    for class_name in CLASS_NAMES:
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Folder does not exist: {class_dir}")

        for img_path in sorted(class_dir.rglob("*"), key=lambda p: str(p).casefold()):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                data.append({"filepath": str(img_path), "label": class_name})
                
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError(f"No images found in {data_dir}.")
    return df


def random_oversampling(train_dataframe):
    """Balances the dataset by oversampling minority classes to match the majority class."""
    max_count = train_dataframe["label"].value_counts().max()
    balanced_parts = []

    for class_name in CLASS_NAMES:
        class_df = train_dataframe[train_dataframe["label"] == class_name]
        if len(class_df) == 0:
            raise ValueError(f"Class {class_name} has no samples.")
            
        class_oversampled = class_df.sample(n=max_count, replace=True, random_state=SEED)
        balanced_parts.append(class_oversampled)

    return pd.concat(balanced_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def grouped_train_val_split(df, group_col="lesion_id", test_size=0.30, seed=42):
    """
    Splits the DataFrame into Train and Validation ensuring that images from the 
    same group (lesion/patient) are not separated across the sets.
    """
    # Check if the grouping column exists in the DataFrame
    if group_col not in df.columns:
        raise ValueError(f"The column '{group_col}' was not found in the DataFrame. "
                         "You need to load the ISIC metadata.")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    
    # gss.split returns index generators. We use next() to get the first (and only) split.
    train_idx, val_idx = next(gss.split(df, groups=df[group_col]))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    return train_df, val_df


def grouped_cv_fold_split(df, fold_index, n_splits=5, group_col="lesion_id", seed=42):
    """Returns one reproducible stratified, group-independent CV fold."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(df, y=df["label"], groups=df[group_col]))
    if fold_index < 0 or fold_index >= len(folds):
        raise ValueError(f"CV_FOLD must be between 0 and {len(folds) - 1}.")
    train_idx, val_idx = folds[fold_index]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)



def atomic_json(path, payload):
    """Writes a complete JSON before replacing its previous version."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def population_fingerprint(df):
    # Verifies identity, class, grouping, and file size/date changes.
    # It is not a visual equality or pixel-by-pixel content audit.
    records = []
    for row in df.sort_values("image").itertuples(index=False):
        stat = Path(row.filepath).stat()
        records.append([str(row.image), str(row.label), str(row.lesion_id),
                        stat.st_size, stat.st_mtime_ns])
    return hashlib.sha256(json.dumps(records, ensure_ascii=False).encode()).hexdigest()


def fixed_holdout(development, test):
    """Creates a single split and reuses its images across ALL executions."""
    development, test = development.copy(), test.copy()
    for df, name in [(development, "development"), (test, "test")]:
        df["image"] = df["filepath"].map(lambda p: Path(p).stem)
        if "lesion_id" not in df:
            df["lesion_id"] = df["image"]
        df["lesion_id"] = df["lesion_id"].fillna(df["image"]).astype(str)
        blank = df["lesion_id"].str.strip().eq("")
        df.loc[blank, "lesion_id"] = df.loc[blank, "image"]
        if df["image"].str.casefold().duplicated().any():
            raise ValueError(f"Duplicated image identifiers in {name}.")
        if set(df["label"]) != set(CLASS_NAMES):
            raise ValueError(f"Missing or unexpected classes in {name}.")
    overlap = set(development.image.str.casefold()) & set(test.image.str.casefold())
    if overlap:
        raise ValueError(f"There are shared images between development and test: {sorted(overlap)[:8]}")
    development = development.sort_values("image").reset_index(drop=True)
    test = test.sort_values("image").reset_index(drop=True)
    fixed_path = Path(OUTPUT_ROOT) / "_fixed_split" / "fixed_split.json"
    expected = {
        "schema": 1, "evaluation_protocol": "fixed_holdout", "split_seed": SPLIT_SEED,
        "validation_fraction_groups": VALIDATION_FRACTION, "classes": CLASS_NAMES,
        "development_fingerprint": population_fingerprint(development),
        "test_fingerprint": population_fingerprint(test),
    }
    if fixed_path.exists():
        state = json.loads(fixed_path.read_text(encoding="utf-8"))
        if any(state.get(k) != v for k, v in expected.items()):
            raise ValueError("The data or split configuration changed. "
                             "They will not be mixed with existing results; use another output folder.")
        base = development.set_index("image", drop=False)
        train = base.loc[state["train_images"]].reset_index(drop=True)
        validation = base.loc[state["validation_images"]].reset_index(drop=True)
        if state["test_images"] != test.image.tolist():
            raise ValueError("The test set does not match the saved split.")
        to_hash = {k: v for k, v in state.items() if k != "split_id"}
        if state.get("split_id") != hashlib.sha256(json.dumps(to_hash, sort_keys=True).encode()).hexdigest():
            raise ValueError("The fixed split record was modified.")
    else:
        train, validation = grouped_train_val_split(
            development, group_col="lesion_id", test_size=VALIDATION_FRACTION, seed=SPLIT_SEED)
        state = dict(expected, train_images=train.image.tolist(),
                     validation_images=validation.image.tolist(), test_images=test.image.tolist())
        state["split_id"] = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    if set(train.image) & set(validation.image):
        raise ValueError("There are shared images between train and validation.")
    if set(train.lesion_id) & set(validation.lesion_id):
        raise ValueError("There are shared lesion groups between train and validation.")
    if set(train.image) | set(validation.image) != set(development.image):
        raise ValueError("Train and validation do not cover the development set.")
    for name, df in [("train", train), ("validation", validation)]:
        if set(df.label) != set(CLASS_NAMES):
            raise ValueError(f"The {name} split does not contain all three classes.")
    if not fixed_path.exists():
        atomic_json(fixed_path, state)
    print("\nFIXED SPLIT (without CV):", state["split_id"])
    print(pd.DataFrame({"train": train.label.value_counts(),
                        "validation": validation.label.value_counts(),
                        "test": test.label.value_counts()}).fillna(0).astype(int).to_string())
    return train, validation, test, state["split_id"]


# =============================================================================
# 3. DATA PROCESSING & DATASETS
# =============================================================================

class ApplyCLAHE(object):
    """Custom PyTorch transform to apply Contrast Limited Adaptive Histogram Equalization."""
    def __init__(self, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE_GRID):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        l_clahe = clahe.apply(l)
        
        lab_clahe = cv2.merge((l_clahe, a, b))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_clahe)


class TempDataset(Dataset):
    """Temporary dataset used strictly for calculating mean and standard deviation."""
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            ApplyCLAHE(),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.dataframe)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.dataframe.loc[idx, "filepath"]).convert("RGB"))


def get_mean_std(train_df):
    """Calculates exact mean and standard deviation of the training dataset for normalization."""
    print("\nCalculating exact Mean and Standard Deviation of the training set...")
    temp_loader = DataLoader(TempDataset(train_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    pixel_count = 0
    sum_pixels = torch.zeros(3)
    sum_sq_pixels = torch.zeros(3)
    
    for images in temp_loader:
        b = images.size(0)
        sum_pixels += images.sum([0, 2, 3])
        sum_sq_pixels += (images ** 2).sum([0, 2, 3])
        pixel_count += b * images.size(2) * images.size(3)
        
    mean = sum_pixels / pixel_count
    variance = torch.clamp((sum_sq_pixels / pixel_count) - (mean ** 2), min=0.0)
    std = torch.sqrt(variance)
    
    print(f"Mean: {mean.tolist()} | Std: {std.tolist()}")
    return mean.tolist(), std.tolist()


class SkinCancerDatasetRAM(Dataset):
    """
    Highly optimized Dataset that preloads all unique resized images into RAM.
    Eliminates disk I/O bottlenecks during model training.
    """
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_dict = {}
        
        # OPTIMIZATION: Convert columns to native lists for __getitem__
        self.filepaths = self.dataframe["filepath"].values
        self.labels = [label_to_index[lbl] for lbl in self.dataframe["label"].values]        
        
        unique_filepaths = self.dataframe["filepath"].unique()
        print(f"\nPreloading {len(unique_filepaths)} UNIQUE images into RAM...")
        preload_transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            ApplyCLAHE(), # Apply CLAHE just one time
            transforms.ToTensor()
        ])
        
        for img_path in unique_filepaths:
            self.image_dict[img_path] = preload_transform(Image.open(img_path).convert("RGB"))
                
    def __len__(self): return len(self.filepaths)
    def __getitem__(self, idx):
        # Access is now immediate (O(1)), without going through the Pandas engine.
        img_path = self.filepaths[idx]
        label = self.labels[idx]
        img = self.image_dict[img_path].clone()
        
        if self.transform:
            img = self.transform(img)
        return img, label


class SkinCancerDatasetDisk(Dataset):
    """Standard Dataset loading images from Disk. Used for Testing to save RAM."""
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

        # OPTIMIZATION: Convert columns to native lists for __getitem__
        self.filepaths = self.dataframe["filepath"].values
        self.labels = [label_to_index[lbl] for lbl in self.dataframe["label"].values]

    def __len__(self):  return len(self.filepaths) 
    def __getitem__(self, idx):
        # Access is now immediate (O(1)), without going through the Pandas engine.
        img_path = self.filepaths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        return image, label       


class IndexedSubsetDataset(Dataset):
    """Wraps a dataset to only expose a dynamically chosen subset of indices (used for Hard Mining)."""
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.base_dataset[int(self.indices[idx])]


# =============================================================================
# 4. NEURAL NETWORK ARCHITECTURE
# =============================================================================

class DCNN_ISIC2019_2(nn.Module):
    """
    Deep Convolutional Neural Network designed for ISIC2019.
    Utilizes progressive channel expansion and intensive Dropout/BatchNorm for regularization.
    """
    def __init__(self, num_classes=8):
        super(DCNN_ISIC2019_2, self).__init__()
        
        # Feature Extractor (Convolutional Blocks)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )
        
        # Final Convolutional layers (separated for easy Grad-CAM hooking)
        self.last_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.last_conv_act = nn.Sequential(
            nn.ReLU(inplace=False), 
            nn.BatchNorm2d(256)
        )
        
        self.pool = nn.AdaptiveAvgPool2d(4) 
        
        # Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, num_classes) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.last_conv(x)
        x = self.last_conv_act(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class NASNet_ISIC2019(nn.Module):
    """
    NASNet Model with Transfer Learning and Warm-up control.
    """
    def __init__(self, num_classes=8, pretrained=True):
        super(NASNet_ISIC2019, self).__init__()
        
        self.backbone = timm.create_model('mnasnet_100', pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        
        #self.classifier = nn.Sequential(
        #    nn.Dropout(0.5),
        #    nn.Linear(in_features, 256),
        #    nn.ReLU(inplace=True),
            #nn.BatchNorm1d(256),
            
            #nn.Linear(256, 128),
           #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(128),
            
            #nn.Linear(128, 64),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(64),
            
        #    nn.Linear(256, num_classes) 
        #)
        
        self.classifier = nn.Sequential(
    	    nn.Dropout(p=0.5),
    	    nn.Linear(in_features, 64),
    	    nn.BatchNorm1d(64),
    	    nn.SiLU(), # SiLU/Swish usually performs better than ReLU in modern heads
    	    nn.Dropout(p=0.2),
    	    nn.Linear(64, num_classes)
	)

    def forward(self, x):
        return self.classifier(self.backbone(x))
        
    def freeze_backbone(self):
        """Freezes the backbone weights (used in warm-up)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreezes the backbone weights for Fine-Tuning the entire network"""
        for param in self.backbone.parameters():
            param.requires_grad = True


# =============================================================================
# 5. TRAINING UTILITIES, LOSS & METRICS
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss to down-weight easy examples and focus on hard, misclassified ones."""
    def __init__(self, alpha=None, gamma=FOCAL_LOSS_GAMMA, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_device = self.alpha.to(inputs.device)
            focal_loss = alpha_device.gather(0, targets.data) * focal_loss
            
        if self.reduction == 'mean': return focal_loss.mean()
        if self.reduction == 'sum': return focal_loss.sum()
        return focal_loss


def rand_bbox(size, lam):
    """Generates a random bounding box for CutMix Data Augmentation."""
    W, H = size[3], size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)

    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def stable_softmax_weights(losses, temperature=1.0):
    """Computes stable probabilities via Softmax for Hard-Mining sample selection."""
    x = np.asarray(losses, dtype=np.float64) / max(float(temperature), 1e-8)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def compute_sample_losses(model, dataset_full, indices, device, batch_size):
    """Evaluates the model on current indices without gradients to find 'hard' samples."""
    subset_dataset = IndexedSubsetDataset(dataset_full, indices)
    loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    model.eval()
    criterion_none = FocalLoss(reduction='none').to(device)
    all_labels = dataset_full.dataframe["label"].values
    
    losses_list = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss = criterion_none(model(inputs), labels)
        
            losses_list.append(loss.float().cpu())

    all_losses = torch.cat(losses_list).numpy()
    
    rows = []
    for j, real_idx in enumerate(indices):
        rows.append({
            "real_idx": int(real_idx),
            "label": all_labels[real_idx],
            "sample_loss": float(all_losses[j])
        })

    return pd.DataFrame(rows)


def select_next_epoch_indices(loss_df, drop_rate=DROP_RATE, temperature=HARD_MINING_TEMPERATURE, seed=SEED):
    """Selects a subset of indices for the next epoch, favoring those with higher loss."""
    rng = np.random.default_rng(seed)
    chosen_indices = []

    for class_name in CLASS_NAMES:
        class_df = loss_df[loss_df["label"] == class_name].copy()
        if class_df.empty: continue

        keep_n = max(MIN_KEEP_PER_CLASS, int(round(len(class_df) * (1.0 - drop_rate))))
        weights = stable_softmax_weights(class_df["sample_loss"].to_numpy(), temperature=temperature)
        
        chosen = rng.choice(
            class_df["real_idx"].to_numpy(dtype=np.int64),
            size=keep_n,
            replace=(keep_n > len(class_df)),
            p=weights
        )
        chosen_indices.extend(chosen.tolist())

    rng.shuffle(chosen_indices)
    return np.asarray(chosen_indices, dtype=np.int64)


def select_random_indices(loss_df, drop_rate=DROP_RATE, seed=SEED):
    """Stratified random control with exactly the same retained size as hard mining."""
    rng = np.random.default_rng(seed)
    chosen_indices = []
    for class_name in CLASS_NAMES:
        class_df = loss_df[loss_df["label"] == class_name]
        if class_df.empty:
            continue
        keep_n = max(MIN_KEEP_PER_CLASS, int(round(len(class_df) * (1.0 - drop_rate))))
        chosen = rng.choice(class_df["real_idx"].to_numpy(dtype=np.int64), size=keep_n,
                            replace=(keep_n > len(class_df)))
        chosen_indices.extend(chosen.tolist())
    rng.shuffle(chosen_indices)
    return np.asarray(chosen_indices, dtype=np.int64)


def evaluate_mean_loss(model, dataset_full, indices, criterion, device, batch_size):
    """Helper to calculate the true average loss over an entire dataset split."""
    loader = DataLoader(IndexedSubsetDataset(dataset_full, indices), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    total_loss, total_samples = 0.0, 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss = criterion(model(inputs), labels)
                
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
    return total_loss / max(total_samples, 1)


# =============================================================================
# 6. EVALUATION & VISUALIZATION
# =============================================================================

def plot_training_curves(history_df, output_dir):
    """Saves Accuracy and Loss line charts over training epochs."""
    # Accuracy Plot
    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["accuracy"], label="Training")
    plt.plot(history_df["epoch"], history_df["val_accuracy"], label="Validation")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Loss Plot
    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["loss"], label="Training")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()


def multiclass_specificity(y_true, y_pred, num_classes):
    """Calculates Specificity (True Negative Rate) for multiclass classification."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

    return np.mean(specificities), specificities


def make_gradcam_heatmap(img_tensor, model, pred_index=None):
    """Generates a Grad-CAM heatmap to visualize network attention regions."""
    gradients, activations = [], []
    
    def fwd_hook(module, input, output): activations.append(output)
    def bwd_hook(module, grad_input, grad_output): gradients.append(grad_output[0])
        
    handle_fwd = model.last_conv.register_forward_hook(fwd_hook)
    handle_bwd = model.last_conv.register_full_backward_hook(bwd_hook)

    model.eval()
    preds = model(img_tensor)
    if pred_index is None:
        pred_index = torch.argmax(preds, dim=1).item()
        
    model.zero_grad()
    preds[0, pred_index].backward()
    
    handle_fwd.remove()
    handle_bwd.remove()
    
    grads, acts = gradients[0].cpu().data.numpy()[0], activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    
    heatmap = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        heatmap += w * acts[i, :, :]
        
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    
    return cv2.resize(heatmap, (img_tensor.size(3), img_tensor.size(2))), preds


# =============================================================================
# 7. MAIN EXECUTION (ENTRY POINT)
# =============================================================================

if __name__ == '__main__':
    print(f"Using device: {device}")
    
        # --- WANDB INITIALIZATION ---
    if USE_WANDB:
    # Automatic login if the key is already configured in the terminal,
    # otherwise it will ask for a link on the first run.
        wandb.init(
            project=WANDB_PROJECT,
            name=f"{chosen_model}_{SELECTION_METHOD}_Drop{DROP_RATE}_Seed{SEED}_Fold{CV_FOLD + 1 if CV_FOLD >= 0 else 'Final'}",
            config={
                "model": chosen_model,
                "img_size": IMG_SIZE[0],
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LR,
                "focal_gamma": FOCAL_LOSS_GAMMA,
                "drop_rate": DROP_RATE,
                "seed": SEED,
                "split_seed": SPLIT_SEED,
                "selection_method": SELECTION_METHOD,
                "cv_fold": CV_FOLD,
                "augmentation_prob": AUGMENTATION_PROB,
                "hard_mining_temp": HARD_MINING_TEMPERATURE,
                "use_random_oversampling": USE_RANDOM_OVERSAMPLING
            }
        )

    # --- 1. Load DataFrames ---
    train_val_df = build_dataframe(DATA_DIR)
    test_df = build_dataframe(DATA_TEST_DIR)

    # --- 2. Train/Val Split & Balancing ---
    #train_pure_df, val_df = train_test_split(
    #    train_val_df, test_size=0.20, stratify=train_val_df["label"], random_state=SEED
    #)
    
    #Split grouped images by patient
    metadata_df = pd.read_csv(metadata_path)
    #metadata_df = pd.read_csv(metadata_path, sep=";")
    
    # 2. Extract the image name from the filepath to perform the merge
    # Assuming the filename is "ISIC_0000001.jpg", we get the base name without the extension
    train_val_df['image'] = train_val_df['filepath'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    
    # 3. Merge the data to bring the 'lesion_id' column into your DataFrame
    train_val_df = pd.merge(train_val_df, metadata_df[['image', 'lesion_id']], on='image', how='left', validate='many_to_one')
    
    # Fill null lesions (some ISIC images don't have a lesion_id, so each becomes a unique group)
    train_val_df['lesion_id'] = train_val_df['lesion_id'].fillna(train_val_df['image'])
    
    # A single split shared across all rates, methods, and seeds.
    train_pure_df, val_df, test_df, split_id = fixed_holdout(train_val_df, test_df)
    train_pure_df.to_csv(os.path.join(OUTPUT_DIR, "train_pure_split.csv"), **CSV_OPTIONS)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "validation_split.csv"), **CSV_OPTIONS)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_split.csv"), **CSV_OPTIONS)
    run_identity = {
        "selection_method": SELECTION_METHOD, "drop_rate": DROP_RATE,
        "seed": SEED, "split_seed": SPLIT_SEED, "evaluation_protocol": "fixed_holdout",
        "validation_fraction_groups": VALIDATION_FRACTION, "split_id": split_id,
        "trainer_sha256": file_digest(__file__),
    }
    atomic_json(Path(OUTPUT_DIR) / "run_configuration.json", run_identity)

    # Apply Random Oversampling
        # Apply Random Oversampling (Optional)
    if USE_RANDOM_OVERSAMPLING:
        print("\n[INFO] Random oversampling is enabled. Balancing the classes...")
        train_final_df = random_oversampling(train_pure_df)
        train_final_df.to_csv(os.path.join(OUTPUT_DIR, "train_final_balanced.csv"), **CSV_OPTIONS)
    else:
        print("\n[INFO] Random oversampling is DISABLED. Using the original distribution of classes.")
        train_final_df = train_pure_df.copy()
        train_final_df.to_csv(os.path.join(OUTPUT_DIR, "train_final_unbalanced.csv"), **CSV_OPTIONS)

    # --- 3. Compute Stats & Build Transforms ---
    TRAIN_MEAN, TRAIN_STD = get_mean_std(train_final_df)

    train_transforms = transforms.Compose([
        #ApplyCLAHE(),
        #transforms.Pad(padding=int(IMG_SIZE[0] * 0.5), padding_mode='reflect'), #added
        transforms.RandomRotation(degrees=45),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
       #transforms.CenterCrop(IMG_SIZE), #added
        #transforms.ToTensor(), 
        transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0) 
    ])

    val_transforms = transforms.Compose([
        #ApplyCLAHE(),
        #transforms.ToTensor(),
        transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD) 
    ]) 

    test_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE), 
        ApplyCLAHE(),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD) 
    ]) 

    # --- 4. Initialize Datasets & Loaders ---
    train_dataset_full = SkinCancerDatasetRAM(train_final_df, transform=train_transforms)
    val_dataset_full = SkinCancerDatasetRAM(val_df, transform=val_transforms)
    test_dataset = SkinCancerDatasetDisk(test_df, transform=test_transforms)
    
    val_loader = DataLoader(val_dataset_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # --- 5. Setup Model, Loss, Optimizer ---
    if chosen_model == "DCNN":
        model = DCNN_ISIC2019_2(num_classes=NUM_CLASSES).to(device)
    if chosen_model == "NasNet":
        model = NASNet_ISIC2019(num_classes=NUM_CLASSES, pretrained=True).to(device)
        # Start the warm-up by freezing the backbone.
        model.freeze_backbone()
        print(f"\n[INFO] WARM-UP ENABLED: Backbone frozen for the first {WARMUP_EPOCHS} epochs.")
    if chosen_model == "EfficientNet":
        if WARMUP_EPOCHS > 0 and EPOCHS == WARMUP_EPOCHS + 1:

            print("\n" + "=" * 50)
            print("[INFO] WARM-UP COMPLETE.")
            print("[INFO] Unfreezing EfficientNet backbone.")

            base_model = (
                model._orig_mod
                if hasattr(model, "_orig_mod")
                else model
            )

            base_model.unfreeze_backbone()

            print("=" * 50 + "\n")

        if (
            WARMUP_EPOCHS > 0
            and WARMUP_EPOCHS < EPOCHS
            <= WARMUP_EPOCHS + RAMP_WARMUP_EPOCHS
        ):
            lr_factor = (
                EPOCHS - WARMUP_EPOCHS
            ) / RAMP_WARMUP_EPOCHS

            current_lr = LR * lr_factor

            for param_group in optimize.param_groups:
                param_group["lr"] = current_lr

            print(
                f"[INFO] Fine-tuning learning rate: "
                f"{current_lr:.8f}"
            )

    #if hasattr(torch, 'compile'):
    #    print("\n[INFO] PyTorch 2.0+ detected. Compiling the model...")
    #    model = torch.compile(model) #This will cause PyTorch to compile the network at a lower level, generating a "free" performance gain of approximately 10% to 30% per epoch!
    #else:
    #    print("\n[INFO] PyTorch < 2.0 detected. Skipping the torch.compile() step.")

    print("\n[INFO] torch.compile disabled for Windows.")
        
    criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA, reduction='mean')
    optimizer = optim.Adamax(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_model_path = os.path.join(OUTPUT_DIR, "best_nasnet_ISIC2019_3classes.pth")
    
    # Trackers for Early Stopping
    best_val_loss_es = float("inf")
    patience_counter = 0
    best_weights_es = None

    all_train_indices = np.arange(len(train_dataset_full), dtype=np.int64)
    current_train_indices = all_train_indices.copy()

    history = {"epoch": [], "accuracy": [], "val_accuracy": [], "loss": [], "val_loss": [],
               "val_f1_macro": [], "val_precision_macro": [], "val_recall_macro": [],
               "val_auc_macro_ovr": [], "elapsed_training_seconds": [],
               "train_full_loss": [], "selected_train_size": []}
    
    sample_loss_df = None 

    # ============================================================
    # 6. TRAINING LOOP (With Dynamic Hard Mining)
    # ============================================================

    train_subset = IndexedSubsetDataset(train_dataset_full, current_train_indices)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    # Measure the complete training stage: training, validation, hard mining,
    # early stopping and checkpoint saving. CUDA synchronization is required
    # because GPU operations are asynchronous.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        
        if chosen_model == "NasNet":
            # --- WARM-UP VERIFICATION ---
            if epoch == WARMUP_EPOCHS + 1:
                print("\n" + "="*50)
                print("[INFO] WARM-UP COMPLETE! Unfreeze the backbone for fine-tuning...")
                base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                base_model.unfreeze_backbone()
                print("="*50 + "\n")
            
            if WARMUP_EPOCHS < epoch <= WARMUP_EPOCHS + RAMP_WARMUP_EPOCHS:
                reduce_LR = (epoch - WARMUP_EPOCHS)/RAMP_WARMUP_EPOCHS
                for param_group in optimizer.param_groups:
                    param_group['lr'] = LR * reduce_LR 
                print(f"[INFO] Learning rate adjusted to Fine-Tuning: {LR * reduce_LR}")
            # ------------------------------
        
        #train_subset = IndexedSubsetDataset(train_dataset_full, current_train_indices)
        #train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        
        model.train()
        train_loss_sum, train_corrects, train_total = 0.0, 0, 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            r = np.random.rand()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                # MixUp Data Augmentation
                if r < (AUGMENTATION_PROB / 2):
                    lam = np.random.beta(1.0, 1.0)
                    rand_index = torch.randperm(inputs.size(0), device=device)
                    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[rand_index]
                    outputs = model(mixed_inputs)
                    loss = lam * criterion(outputs, labels) + (1.0 - lam) * criterion(outputs, labels[rand_index])
                    dominant_target = labels if lam >= 0.5 else labels[rand_index]
                
                # CutMix Data Augmentation
                elif r < AUGMENTATION_PROB:
                    lam = np.random.beta(1.0, 1.0)
                    rand_index = torch.randperm(inputs.size(0), device=device)
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                    mixed_inputs = inputs.clone()
                    mixed_inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[rand_index, :, bby1:bby2, bbx1:bbx2]
                    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(-1) * inputs.size(-2)))
                    outputs = model(mixed_inputs)
                    loss = lam * criterion(outputs, labels) + (1.0 - lam) * criterion(outputs, labels[rand_index])
                    dominant_target = labels if lam >= 0.5 else labels[rand_index]
                
                # Standard Forward Pass
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    dominant_target = labels

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * inputs.size(0)
            train_corrects += torch.sum(outputs.argmax(1) == dominant_target.data).item()
            train_total += labels.size(0)

        epoch_train_loss = train_loss_sum / max(train_total, 1)
        epoch_train_acc = train_corrects / max(train_total, 1)

        # -------------------------
        # Validation Phase
        # -------------------------
        model.eval()
        val_loss_sum, val_corrects, val_total = 0.0, 0, 0
        val_epoch_true, val_epoch_pred, val_epoch_prob = [], [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss_sum += loss.item() * inputs.size(0)
                val_corrects += torch.sum(outputs.argmax(1) == labels.data).item()
                val_total += labels.size(0)
                probabilities = torch.softmax(outputs.float(), dim=1)
                val_epoch_true.extend(labels.cpu().numpy())
                val_epoch_pred.extend(outputs.argmax(1).cpu().numpy())
                val_epoch_prob.extend(probabilities.cpu().numpy())

        epoch_val_loss = val_loss_sum / max(val_total, 1)
        epoch_val_acc = val_corrects / max(val_total, 1)
        epoch_val_f1 = f1_score(val_epoch_true, val_epoch_pred, average="macro", zero_division=0)
        epoch_val_precision = precision_score(val_epoch_true, val_epoch_pred, average="macro", zero_division=0)
        epoch_val_recall = recall_score(val_epoch_true, val_epoch_pred, average="macro", zero_division=0)
        try:
            epoch_val_auc = roc_auc_score(label_binarize(val_epoch_true, classes=list(range(NUM_CLASSES))),
                                          np.asarray(val_epoch_prob), average="macro", multi_class="ovr")
        except ValueError:
            epoch_val_auc = np.nan

        train_full_loss = evaluate_mean_loss(model, train_dataset_full, all_train_indices, criterion, device, BATCH_SIZE) if COMPUTE_FULL_TRAIN_LOSS else epoch_train_loss

        current_lr = optimizer.param_groups[0]['lr']
        elapsed_training_seconds = time.perf_counter() - training_start_time
        print(f"Epoch {epoch}/{EPOCHS} - lr: {current_lr:.6f} - loss: {epoch_train_loss:.4f} - acc: {epoch_train_acc:.4f} - val_loss: {epoch_val_loss:.4f} - val_acc: {epoch_val_acc:.4f} - val_f1: {epoch_val_f1:.4f}")

        # --- WANDB LOGGING ---
        if USE_WANDB:
            wandb.log({
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc,
                "val_f1_macro": epoch_val_f1,
                "val_precision_macro": epoch_val_precision,
                "val_recall_macro": epoch_val_recall,
                "val_auc_macro_ovr": epoch_val_auc,
                "elapsed_training_seconds": elapsed_training_seconds,
                "elapsed_training_minutes": elapsed_training_seconds / 60.0,
                "learning_rate": current_lr,
                "train_pool_size": len(current_train_indices) # To see Hard Mining in action
            })

        # Update History
        history["epoch"].append(epoch)
        history["loss"].append(epoch_train_loss)
        history["accuracy"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)
        history["val_f1_macro"].append(epoch_val_f1)
        history["val_precision_macro"].append(epoch_val_precision)
        history["val_recall_macro"].append(epoch_val_recall)
        history["val_auc_macro_ovr"].append(epoch_val_auc)
        history["elapsed_training_seconds"].append(elapsed_training_seconds)
        history["train_full_loss"].append(train_full_loss)
        history["selected_train_size"].append(len(current_train_indices))


        if chosen_model == "NasNet":
            if epoch > (WARMUP_EPOCHS + RAMP_WARMUP_EPOCHS):
                scheduler.step(epoch_val_loss)
        else:
            scheduler.step(epoch_val_loss)

        # Save Best Model & Early Stopping Logic
        if epoch_val_loss < best_val_loss_es:
            best_val_loss_es = epoch_val_loss
            patience_counter = 0
            #best_weights_es = copy.deepcopy(model.state_dict()) #save memory
            #torch.save(model.state_dict(), best_model_path)
            #torch.save(model._orig_mod.state_dict(), best_model_path)
            base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(base_model.state_dict(), best_model_path)

            print("  -> Val Loss Improved! Model Saved.")
        else:
            patience_counter += 1
            if patience_counter >= ES_PATIENCE:
                print(f"Early Stopping triggered at Epoch {epoch}.")
                base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                base_model.load_state_dict(torch.load(best_model_path))
                #model._orig_mod.load_state_dict(torch.load(best_model_path))
                #model.load_state_dict(best_weights_es)
                break

        # -------------------------
        # Hard Mining Re-Selection
        # -------------------------
        if DROP_RATE > 0:
            # It only updates the sample loss and re-selects the indices every N epochs.
            if SELECTION_METHOD == "random" and sample_loss_df is None:
                sample_loss_df = pd.DataFrame({
                    "real_idx": all_train_indices,
                    "label": train_dataset_full.dataframe["label"].values,
                    "sample_loss": np.zeros(len(all_train_indices), dtype=float)
                })
            elif SELECTION_METHOD == "hard" and (sample_loss_df is None or epoch % UPDATE_LOSS_FREQ == 0):
                print(f"\n[INFO] Updating sample losses for Hard Mining (Epoch {epoch})...")
            
                # Temporarily use validation transforms (no random augmentations) to measure true loss cleanly
                original_transforms = train_dataset_full.transform
                train_dataset_full.transform = val_transforms
            
                sample_loss_df = compute_sample_losses(model, train_dataset_full, all_train_indices, device, BATCH_SIZE)
            
                train_dataset_full.transform = original_transforms  # Restore training transforms
            
            # Draws samples EVERY EPOCH
            # Since the seed is 'SEED + epoch', it will generate a different subset
            # each epoch, even using the old 'sample_loss_df'.
            if SELECTION_METHOD == "hard":
                current_train_indices = select_next_epoch_indices(sample_loss_df, DROP_RATE, HARD_MINING_TEMPERATURE, SEED + epoch)
            elif SELECTION_METHOD == "random":
                current_train_indices = select_random_indices(sample_loss_df, DROP_RATE, SEED + epoch)
            else:
                raise ValueError("SELECTION_METHOD must be 'hard' or 'random'.")
            train_subset = IndexedSubsetDataset(train_dataset_full, current_train_indices)
            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

        else:
            # If DROP_RATE == 0, the pool size will be the same of the full dataset
            current_train_indices = all_train_indices
            
            # Note: We do not need to create again the train_loader, because it was create with shuffle=True

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_end_time = time.perf_counter()

    training_time_seconds = training_end_time - training_start_time
    training_time_minutes = training_time_seconds / 60.0
    training_time_hours = training_time_seconds / 3600.0
    executed_epochs = len(history["epoch"])
    mean_time_per_epoch_seconds = (
        training_time_seconds / max(executed_epochs, 1)
    )

    print(
        f"\nTotal training time: "
        f"{training_time_seconds:.2f} s | "
        f"{training_time_minutes:.2f} min | "
        f"{training_time_hours:.2f} h"
    )

    training_time_df = pd.DataFrame([{
        "executed_epochs": executed_epochs,
        "training_time_seconds": training_time_seconds,
        "training_time_minutes": training_time_minutes,
        "training_time_hours": training_time_hours,
        "mean_time_per_epoch_seconds": mean_time_per_epoch_seconds
    }])
    training_time_df.to_csv(os.path.join(OUTPUT_DIR, "training_time.csv"), **CSV_OPTIONS)

    # Save training curves and history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), **CSV_OPTIONS)
    plot_training_curves(history_df, OUTPUT_DIR)


    # ============================================================
    # 7. FINAL EVALUATION (Deterministic TTA)
    # ============================================================
    print("\nExecuting Test Evaluation with Deterministic TTA (4 views per image)...")
    #model.load_state_dict(torch.load(best_model_path))
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    base_model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Final validation metrics from the same best checkpoint used for test.
    val_true, val_pred, val_prob = [], [], []
    val_best_loss_sum, val_best_total = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(images)
                batch_loss = criterion(outputs, labels)
            probabilities = torch.softmax(outputs.float(), dim=1)
            predictions = probabilities.argmax(1)
            val_best_loss_sum += batch_loss.item() * images.size(0)
            val_best_total += images.size(0)
            val_true.extend(labels.cpu().numpy())
            val_pred.extend(predictions.cpu().numpy())
            val_prob.extend(probabilities.cpu().numpy())

    val_true, val_pred, val_prob = np.asarray(val_true), np.asarray(val_pred), np.asarray(val_prob)
    val_specificity, _ = multiclass_specificity(val_true, val_pred, NUM_CLASSES)
    try:
        val_auc_macro = roc_auc_score(label_binarize(val_true, classes=list(range(NUM_CLASSES))),
                                      val_prob, average="macro", multi_class="ovr")
    except ValueError:
        val_auc_macro = np.nan
    validation_metrics = {
        "selection_method": SELECTION_METHOD, "drop_rate": DROP_RATE,
        "seed": SEED, "split_seed": SPLIT_SEED,
        "cv_fold": CV_FOLD + 1 if CV_FOLD >= 0 else np.nan,
        "accuracy": accuracy_score(val_true, val_pred),
        "precision_macro": precision_score(val_true, val_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(val_true, val_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(val_true, val_pred, average="macro", zero_division=0),
        "specificity_avg": val_specificity, "auc_macro_ovr": val_auc_macro,
        "validation_loss": val_best_loss_sum / max(val_best_total, 1),
        "training_time_seconds": training_time_seconds, "executed_epochs": executed_epochs
    }
    validation_metrics.update(run_identity)
    validation_metrics["training_time_minutes"] = training_time_minutes
    validation_metrics["mean_time_per_epoch_seconds"] = mean_time_per_epoch_seconds
    pd.DataFrame([validation_metrics]).to_csv(os.path.join(OUTPUT_DIR, "validation_metrics.csv"), **CSV_OPTIONS)

    test_loss, test_corrects, test_total = 0.0, 0, 0
    y_true, y_pred, y_prob = [], [], []

    # This measurement includes data loading, host-to-device transfer and the
    # four deterministic TTA forward passes used for each test image.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_start_time = time.perf_counter()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Forward passes for 4 orientations
            #out_orig = model(images)
            #out_hf = model(TF.hflip(images))
            #out_vf = model(TF.vflip(images))
            #out_hvf = model(TF.vflip(TF.hflip(images)))
            
            #out_orig = model(images)
            #out_hf = model(torch.flip(images, dims=[3])) 
            #out_vf = model(torch.flip(images, dims=[2]))
            #out_hvf = model(torch.flip(images, dims=[2, 3])) 
            
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                out_orig = model(images)
                out_hf = model(torch.flip(images, dims=[3]))
                out_vf = model(torch.flip(images, dims=[2]))
                out_hvf = model(torch.flip(images, dims=[2, 3]))
            
            
            # Stack the 4 views on the batch axis (If the batch was 32, it temporarily becomes 128) (If there is enough memory in GPU)
            # images_4x = torch.cat([images, torch.flip(images, dims=[3]), torch.flip(images, dims=[2]), torch.flip(images, dims=[2, 3])], dim=0)
            # with torch.cuda.amp.autocast(enabled=USE_AMP):
            #     out_4x = model(images_4x)
            # out_orig, out_hf, out_vf, out_hvf = torch.chunk(out_4x.float(), 4, dim=0)

            # Average the losses and probabilities across the 4 views
            avg_loss = (criterion(out_orig, labels) + criterion(out_hf, labels) + criterion(out_vf, labels) + criterion(out_hvf, labels)) / 4.0
            test_loss += avg_loss.item() * images.size(0)
            
            prob_avg = (torch.softmax(out_orig.float(), dim=1) + 
            torch.softmax(out_hf.float(), dim=1) + 
            torch.softmax(out_vf.float(), dim=1) + 
            torch.softmax(out_hvf.float(), dim=1)) / 4.0

            preds = prob_avg.argmax(1)
            
            test_corrects += torch.sum(preds == labels.data).item()
            test_total += labels.size(0)
            
            y_prob.extend(prob_avg.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_end_time = time.perf_counter()

    inference_time_seconds = inference_end_time - inference_start_time
    inference_time_minutes = inference_time_seconds / 60.0
    mean_inference_time_seconds = (
        inference_time_seconds / max(test_total, 1)
    )
    mean_inference_time_ms = mean_inference_time_seconds * 1000.0
    throughput_images_second = (
        test_total / inference_time_seconds
        if inference_time_seconds > 0 else 0.0
    )

    print(
        f"\nTotal inference time on test: {inference_time_seconds:.4f} s"
        f"\nMean inference time per image: {mean_inference_time_ms:.4f} ms"
        f"\nThroughput: {throughput_images_second:.2f} images/s"
    )

    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
    
    # --- Calculate & Save Metrics ---
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    specificity_avg, _ = multiclass_specificity(y_true, y_pred, NUM_CLASSES)
    
    try:
        auc_macro = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
        auc_weighted = roc_auc_score(y_true_bin, y_prob, average="weighted", multi_class="ovr")
    except Exception:
        auc_macro, auc_weighted = np.nan, np.nan

    metrics = {
        "selection_method": SELECTION_METHOD,
        "drop_rate": DROP_RATE,
        "seed": SEED,
        "split_seed": SPLIT_SEED,
        "cv_fold": CV_FOLD + 1 if CV_FOLD >= 0 else np.nan,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "specificity_avg": specificity_avg,
        "auc_macro_ovr": auc_macro,
        "auc_weighted_ovr": auc_weighted,
        "test_loss": test_loss / max(test_total, 1),
        "training_time_seconds": training_time_seconds,
        "training_time_minutes": training_time_minutes,
        "training_time_hours": training_time_hours,
        "executed_epochs": executed_epochs,
        "mean_time_per_epoch_seconds": mean_time_per_epoch_seconds,
        "test_images": test_total,
        "inference_time_seconds": inference_time_seconds,
        "inference_time_minutes": inference_time_minutes,
        "mean_inference_time_seconds_per_image": mean_inference_time_seconds,
        "mean_inference_time_ms_per_image": mean_inference_time_ms,
        "throughput_images_second": throughput_images_second,
        "tta_views_per_image": 4
    }
    
    metrics.update(run_identity)
    pd.DataFrame([metrics]).to_csv(os.path.join(OUTPUT_DIR, "test_metrics.csv"), **CSV_OPTIONS)
    print("\nTest Metrics Overview:\n", pd.DataFrame([metrics]).T)

    # Classification Report
    report_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)).T
    report_df.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"), sep=";", decimal=",")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"), sep=";", decimal=",")
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --- WANDB FINAL LOGS & CLOSE ---
    if USE_WANDB:
        # Save the confusion matrix graph in WandB.
        wandb.log({
            "test_accuracy": metrics["accuracy"],
            "test_auc_macro": metrics["auc_macro_ovr"],
            "test_f1_macro": metrics["f1_macro"],
            "training_time_seconds": training_time_seconds,
            "training_time_minutes": training_time_minutes,
            "mean_time_per_epoch_seconds": mean_time_per_epoch_seconds,
            "inference_time_seconds": inference_time_seconds,
            "mean_inference_time_ms_per_image": mean_inference_time_ms,
            "throughput_images_second": throughput_images_second,
            # Now the image exists and can be sent!
            "confusion_matrix_plot": wandb.Image(os.path.join(OUTPUT_DIR, "confusion_matrix.png")) 
        })
        wandb.finish() # Close the connection

    print("\nProcessing complete! All results saved to:", OUTPUT_DIR)

    # Only marks the execution as complete when training and testing have finished.
    required_outputs = ["validation_metrics.csv", "test_metrics.csv", "training_time.csv",
                        "classification_report.csv", "confusion_matrix.csv",
                        "train_pure_split.csv", "validation_split.csv", "test_split.csv"]
    completion = dict(run_identity)
    completion["artifacts_sha256"] = {
        name: file_digest(Path(OUTPUT_DIR) / name) for name in required_outputs
    }
    atomic_json(Path(OUTPUT_DIR) / "run_completed.json", completion)
