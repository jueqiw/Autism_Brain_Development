from fileinput import filename
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import KFold
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import cv2

# ------------------ Normalization Functions ------------------
mid = 1.0
scale = 0.15
def preprocess_jacobian(jacobian):
    return (jacobian - mid) / scale
def postprocess_jacobian(jacobian_norm):
    return jacobian_norm * scale + mid

def compute_sobel_edge(image_np):
    # image_np: (192, 192) numpy array
    sobelx = cv2.Sobel(image_np, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image_np, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mag = (sobel_mag - sobel_mag.min()) / (sobel_mag.max() - sobel_mag.min() + 1e-8)
    return sobel_mag.astype(np.float32)


# ------------------ Dataset ------------------
class MRIDataset(Dataset):
    def __init__(self, original_imgs, transformed_imgs, jacobian_maps):
        self.original = original_imgs
        self.transformed = transformed_imgs
        self.jacobians = jacobian_maps

    def __len__(self):
        return len(self.original)

    def __getitem__(self, idx):
        orig = self.original[idx][0]  # shape (192, 192)
        trans = self.transformed[idx][0]

        sobel_orig = compute_sobel_edge(orig)
        sobel_trans = compute_sobel_edge(trans)

        # Stack: original, transformed, sobel(original), sobel(trans)
        orig_tensor = torch.from_numpy(orig).unsqueeze(0).float()
        trans_tensor = torch.from_numpy(trans).unsqueeze(0).float()
        sobel_orig_tensor = torch.from_numpy(sobel_orig).unsqueeze(0).float()
        sobel_trans_tensor = torch.from_numpy(sobel_trans).unsqueeze(0).float()

        input_pair = torch.cat([orig_tensor, trans_tensor, sobel_orig_tensor, sobel_trans_tensor], dim=0)

        jacobian = self.jacobians[idx]  # shape (192, 192)
        jacobian = preprocess_jacobian(jacobian)  # normalize to [-1, 1]
        jacobian = torch.from_numpy(jacobian).float().unsqueeze(0)

        return input_pair, jacobian
    
# Reparameterization trick
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# ------------------ Encoder ------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=16, ft_bank_baseline=32, dropout_alpha=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(4, ft_bank_baseline, kernel_size=3, padding=1) #change into 4 input channels: orig, trans, sobel(orig), sobel(trans)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(ft_bank_baseline, ft_bank_baseline*2, 3, padding=1)
        self.conv3 = nn.Conv2d(ft_bank_baseline*2, ft_bank_baseline*4, 3, padding=1)
        self.flatten_size = (192 // 8) * (192 // 8) * ft_bank_baseline * 4

        self.dropout = nn.Dropout(dropout_alpha)
        self.fc1 = nn.Linear(self.flatten_size, latent_dim*4)
        self.fc_mean = nn.Linear(latent_dim*4, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim*4, latent_dim*4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.tanh(self.fc1(x))
        return self.fc_mean(x)
        '''
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        z = reparameterize(mean, logvar)
        return mean, logvar, z '''

#REMOVED GENERATOR

# ------------------ Decoder ------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim=16, ft_bank_baseline=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.ft_bank_baseline = ft_bank_baseline
        self.flat_size = (192 // 8) * (192 // 8) * ft_bank_baseline * 4

        self.fc1 = nn.Linear(latent_dim, latent_dim*2)
        self.fc2 = nn.Linear(latent_dim*2, latent_dim*4)
        self.fc3 = nn.Linear(latent_dim*4, self.flat_size)

        self.conv1 = nn.Conv2d(ft_bank_baseline*4, ft_bank_baseline*4, 3, padding=1)
        self.conv2 = nn.Conv2d(ft_bank_baseline*4, ft_bank_baseline*2, 3, padding=1)
        self.conv3 = nn.Conv2d(ft_bank_baseline*2, ft_bank_baseline, 3, padding=1)
        #self.conv4 = nn.Conv2d(ft_bank_baseline, 1, 3, padding=1) #output 1 channel = jacobian value

        self.decoder_tail = nn.Sequential(
            nn.Conv2d(ft_bank_baseline, 1, 3, padding=1),
            nn.Tanh()
        )

        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, z):
        x = torch.tanh(self.fc1(z))
        x = torch.tanh(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, self.ft_bank_baseline*4, 192//8, 192//8)

        x = F.relu(self.conv1(x))
        x = self.upsample(x)
        x = F.relu(self.conv2(x))
        x = self.upsample(x)
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = self.decoder_tail(x) #includes conv2d + tanh
        return x


def normalize_for_ssim(x): #normalize between [0, 1] only for loss computation
    min_val = x.amin(dim=[1,2,3], keepdim=True)
    max_val = x.amax(dim=[1,2,3], keepdim=True)
    return (x - min_val) / (max_val - min_val + 1e-8)


# ------------------ Loss ------------------
def vae_weighted_l1_loss(gt, pred, roi_weight=10.0, background_weight=1.0):
    """
    gt, pred, mask: shape (batch, 1, H, W)
    mask: binary mask with 1 for ROI, 0 for background
    """
    mask = (gt != 1).float()  # ROI = where gt is not 1
    weights = torch.where(mask == 1, roi_weight, background_weight)
    l1 = torch.abs(pred - gt)
    weighted_l1 = weights * l1
    return weighted_l1.mean()


# ------------------ Training ------------------
def train_model(train_loader, val_loader, encoder, decoder, epochs=100, lr=1e-3):
    encoder.train()
    decoder.train()

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        encoder.train()
        decoder.train()

        # Training loop
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            z = encoder(x)
            output = decoder(z)
            loss = vae_weighted_l1_loss(y, output)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation loop
        encoder.eval()
        decoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                z = encoder(x)
                output = decoder(z)
                val_loss = vae_weighted_l1_loss(y, output)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    # Plot both training and validation loss
    plt.figure(figsize=(8,5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("pretrain_model/outputs/weighted_l1/all_loss_curve.png", dpi=300)

    return encoder, decoder



# ------------------ Save Outputs ------------------
def save_output_tensor(img_tensor, filename):
    img_np = img_tensor.squeeze().cpu().numpy()
    np.savez_compressed(filename, axial=img_np) #key = axial, change to coronal/sagittal if needed
    '''
    # Min-max normalization for saving
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_uint8 = (img_np * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(filename)

    img_np = img_tensor.squeeze().cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    img_uint8 = (img_np * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(filename)'''

# Loading all data (same logic as TF version)
def get_ages(sub_id, dataset_num):
    sub_id = int(sub_id)
    if dataset_num == 1:
        df = pd.read_csv('../../ace-ig/ABIDE/Phenotypic_V1_0b.csv')
        long_df = pd.read_csv('../../ace-ig/ABIDE/ABIDEII_Long_Composite_Phenotypic.csv') #longitudinal subjects
        if not long_df[(long_df['SUB_ID'] == sub_id)].empty: #subject found in ABIDE II longitudinal subjects - remove
            return None
        age = df[(df['SUB_ID'] == sub_id)]['AGE_AT_SCAN'].values[0]
        if age > 21: return None
    elif dataset_num == 2:
        df = pd.read_csv('../../ace-ig/ABIDE/ABIDEII_Composite_Phenotypic.csv', encoding='cp1252')
        if df[(df['SUB_ID'] == sub_id)].empty:
            return None
        age = df[(df['SUB_ID'] == sub_id)]['AGE_AT_SCAN '].values[0]  # note extra space
        if age > 21: return None
    return age

# Normalize a single slice
def normalize_slice(slice_data):
    lower, upper = np.percentile(slice_data, [0.5, 99.5])
    slice_clipped = np.clip(slice_data, lower, upper)
    normalized = (slice_clipped - lower) / (upper - lower + 1e-8)
    return normalized

def load_data():
    folder_paths = [
        "/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/axial/original",
        #"/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/coronal/original",
        #"/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/sagittal/original",
        "/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/axial/transformed",
        #"/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/coronal/transformed",
        #"/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/sagittal/transformed",
        "/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/axial/jacobian",
        #"/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/coronal/jacobian",
        #"/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/sagittal/jacobian",
        
        "/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/axial/original",
        #"/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/coronal/original",
        #"/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/sagittal/original",
        "/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/axial/transformed",
        #"/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/coronal/transformed",
        #"/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/sagittal/transformed",
        "/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/axial/jacobian",
        #"/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/coronal/jacobian",
        #"/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/sagittal/jacobian",
    ]
    original_list = []
    transformed_list = []
    jacobian_list = []

    for i in range(0, len(folder_paths), 3):
        original_folder = folder_paths[i]
        transformed_folder = folder_paths[i + 1]
        jacobian_folder = folder_paths[i + 2]
        dataset_num = 1 if "ABIDE_I_2D" in original_folder else 2

        print(f"Processing ABIDE {dataset_num} folders.")

        for filename in os.listdir(original_folder):
            if not filename.endswith('.npz'): continue

            # Load original, transformed, and jacobian images
            original_path = os.path.join(original_folder, filename)
            transformed_filename = filename.replace("original", "transformed")
            jacobian_filename = filename.replace("original", "jacobian")
            transformed_path = os.path.join(transformed_folder, transformed_filename)
            jacobian_path = os.path.join(jacobian_folder, jacobian_filename)

            if not (os.path.exists(transformed_path) and os.path.exists(jacobian_path)):
                continue  # skip if anything is missing for that subject

            # Load original npz (assumed key: 'axial')
            original_data = np.load(original_path)
            original_img = original_data["axial"].astype(np.float32)
            original_img = cv2.resize(original_img, (192, 192), interpolation=cv2.INTER_LINEAR)

            # Load transformed npz (assumed key: 'axial')
            transformed_data = np.load(transformed_path)
            transformed_img = transformed_data["axial"].astype(np.float32)
            transformed_img = cv2.resize(transformed_img, (192, 192), interpolation=cv2.INTER_LINEAR)

            # Load jacobian npz (assumed key: 'axial')
            jacobian_data = np.load(jacobian_path)
            jacobian_img = jacobian_data["axial"].astype(np.float32)
            jacobian_img = cv2.resize(jacobian_img, (192, 192), interpolation=cv2.INTER_LINEAR)

            #original_img = normalize_slice(original_img)
            #transformed_img = normalize_slice(transformed_img)
            #jacobian_img = normalize_slice(jacobian_img)

            # Add channels
            original_list.append(original_img[np.newaxis, :, :])
            transformed_list.append(transformed_img[np.newaxis, :, :])
            jacobian_list.append(jacobian_img)

    original_array = np.array(original_list, dtype=np.float32)
    transformed_array = np.array(transformed_list, dtype=np.float32)
    jacobian_array = np.array(jacobian_list, dtype=np.float32)

    print(f"Loaded {original_array.shape[0]} valid samples.")
    return original_array, transformed_array, jacobian_array


# ------------------ Main ------------------
def main():
    # Load preprocessed numpy arrays
    # Shape: (N, 1, 192, 192)
    original, transformed, jacobians = load_data()
    print(f"Original shape: {original.shape}")
    print(f"Transformed shape: {transformed.shape}")
    print(f"Jacobian shape: {jacobians.shape}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(original)):
        print(f"Fold {fold+1}/5")
        train_set = MRIDataset(original[train_idx], transformed[train_idx], jacobians[train_idx])
        val_set = MRIDataset(original[val_idx], transformed[val_idx], jacobians[val_idx])

        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

        encoder = Encoder().to(device)
        decoder = Decoder().to(device)

        encoder, decoder = train_model(train_loader, val_loader, encoder, decoder)

        torch.save(encoder.state_dict(), f"pretrain_model/outputs/weighted_l1/encoder_fold{fold+1}.pt")
        torch.save(decoder.state_dict(), f"pretrain_model/outputs/weighted_l1/decoder_fold{fold+1}.pt")

        # Save some predictions
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x = x.to(device)
                z = encoder(x)
                preds = decoder(z)
                preds_post = postprocess_jacobian(preds)  # back to physical domain
                y_post = postprocess_jacobian(y)
                for j in range(min(3, preds.size(0))):
                    save_output_tensor(preds_post[j], f"pretrain_model/outputs/weighted_l1/fold{fold+1}_val{j}_pred.npz")
                    save_output_tensor(y_post[j], f"pretrain_model/outputs/weighted_l1/fold{fold+1}_val{j}_gt.npz")
                break


if __name__ == "__main__":
    main()