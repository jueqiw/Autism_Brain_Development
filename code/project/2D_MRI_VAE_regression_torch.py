import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from monai.transforms import RandAffine

import scipy.ndimage

torch.set_num_threads(8)  # or the number you request
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalize a single slice
def normalize_slice(slice_data):
    lower, upper = np.percentile(slice_data, [0.5, 99.5])
    slice_clipped = np.clip(slice_data, lower, upper)
    normalized = (slice_clipped - lower) / (upper - lower + 1e-8)
    return normalized

train_transform = RandAffine(
        prob=1.0,
        rotate_range=[np.pi/36],  # 5 degrees
        scale_range=[0.05], #[0.95, 1.05]
        mode='bilinear',
        padding_mode='zeros'
    )

# Dataset class for MRI 2D slices and ages
class MRIDataset(Dataset):
    def __init__(self, image_list, age_list, transform=None):
        self.images = image_list
        self.ages = age_list
        self.transform = transform

    def __len__(self):
        return len(self.ages)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32)  # already normalized numpy array
        img_tensor = torch.from_numpy(img) #shape (1, 192, 192) 
        age = np.array([self.ages[idx]], dtype=np.float32)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, torch.from_numpy(age)

# Reparameterization trick
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# Encoder module
class Encoder(nn.Module):
    def __init__(self, latent_dim=16, ft_bank_baseline=16, dropout_alpha=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, ft_bank_baseline, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(ft_bank_baseline, ft_bank_baseline*2, 3, padding=1)
        self.conv3 = nn.Conv2d(ft_bank_baseline*2, ft_bank_baseline*4, 3, padding=1)
        self.flatten_size = (192 // 8) * (192 // 8) * ft_bank_baseline * 4

        self.dropout = nn.Dropout(dropout_alpha)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flatten_size, latent_dim*4)
        self.fc_z_mean = nn.Linear(latent_dim*4, latent_dim)
        self.fc_z_log_var = nn.Linear(latent_dim*4, latent_dim)

        self.fc_r_mean = nn.Linear(latent_dim*4, 1)
        self.fc_r_log_var = nn.Linear(latent_dim*4, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = torch.tanh(self.fc1(x))

        z_mean = self.fc_z_mean(x)
        z_log_var = self.fc_z_log_var(x)
        r_mean = self.fc_r_mean(x)
        r_log_var = self.fc_r_log_var(x)

        z = reparameterize(z_mean, z_log_var)
        r = reparameterize(r_mean, r_log_var)
        return z_mean, z_log_var, z, r_mean, r_log_var, r

# Generator module (generates latent distribution parameters from age)
class Generator(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.pz_mean = nn.Linear(1, latent_dim)
        self.pz_log_var = nn.Linear(1, 1)

    def forward(self, r):
        pz_mean = self.pz_mean(r)
        pz_log_var = self.pz_log_var(r)
        return pz_mean, pz_log_var

# Decoder module (decodes latent z to image)
class Decoder(nn.Module):
    def __init__(self, latent_dim=16, ft_bank_baseline=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.ft_bank_baseline = ft_bank_baseline

        downsampled_H = 192 // 8
        downsampled_W = 192 // 8
        self.flattened_size = downsampled_H * downsampled_W * ft_bank_baseline * 4

        self.fc1 = nn.Linear(latent_dim, latent_dim*2)
        self.fc2 = nn.Linear(latent_dim*2, latent_dim*4)
        self.fc3 = nn.Linear(latent_dim*4, self.flattened_size)

        self.conv1 = nn.Conv2d(ft_bank_baseline*4, ft_bank_baseline*4, 3, padding=1)
        self.conv2 = nn.Conv2d(ft_bank_baseline*4, ft_bank_baseline*2, 3, padding=1)
        self.conv3 = nn.Conv2d(ft_bank_baseline*2, ft_bank_baseline, 3, padding=1)
        self.conv4 = nn.Conv2d(ft_bank_baseline, 1, 3, padding=1)

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
        x = self.conv4(x)
        # No activation if not binary image (can add sigmoid if needed)
        return x

# Loss function combining reconstruction, KL, and label (age) loss
def vae_loss_fn(x, x_recon, z_mean, z_log_var, pz_mean, pz_log_var, r_mean, r_log_var, r):
    # Reconstruction loss (MAE)
    recon_loss = F.l1_loss(x_recon, x, reduction='mean')

    # KL divergence between posterior z and prior pz
    kl_loss = 1 + z_log_var - pz_log_var - ((z_mean - pz_mean).pow(2) / pz_log_var.exp()) - (z_log_var.exp() / pz_log_var.exp())
    kl_loss = -0.5 * kl_loss.sum(dim=1).mean()

    # Label loss (age prediction)
    label_loss = 0.5 * ((r_mean - r).pow(2) / r_log_var.exp()) + 0.5 * r_log_var
    label_loss = label_loss.mean()

    return recon_loss + kl_loss + label_loss

# Augmentations (same as TF version)
def augment_by_transformation(data, age, n):
    import scipy.ndimage
    if n <= data.shape[0]:
        return data, age
    raw_n = data.shape[0]
    m = n - raw_n
    data_list = [data]
    age_list = [age]
    for _ in range(m):
        idx = np.random.randint(0, raw_n)
        new_data = data[idx].copy()
        new_data = np.squeeze(new_data, axis=0)
        new_data = scipy.ndimage.rotate(new_data, np.random.uniform(-1, 1), axes=(1, 0), reshape=False)
        new_data = scipy.ndimage.rotate(new_data, np.random.uniform(-1, 1), axes=(0, 1), reshape=False)
        new_data = scipy.ndimage.shift(new_data, np.random.uniform(-1, 1))
        new_data = np.expand_dims(new_data, axis=0)
        data_list.append(new_data)
        age_list.append(age[idx])
    new_data = np.concatenate(data_list, axis=0)
    new_age = np.array(age_list)
    return new_data, new_age

# Loading all data (same logic as TF version)
def get_ages(sub_id, dataset_num):
    sub_id = int(sub_id)
    if dataset_num == 1:
        df = pd.read_csv('../../ace-ig/ABIDE/Phenotypic_V1_0b.csv')
        age = df[(df['SUB_ID'] == sub_id)]['AGE_AT_SCAN'].values[0]
        if age > 21: return None
    elif dataset_num == 2:
        df = pd.read_csv('../../ace-ig/ABIDE/ABIDEII_Composite_Phenotypic.csv', encoding='cp1252')
        if df[(df['SUB_ID'] == sub_id)].empty: #
            return None
        age = df[(df['SUB_ID'] == sub_id)]['AGE_AT_SCAN '].values[0]  # note extra space
        if age > 21: return None
    return age

def load_data():
    folder_paths = [
        "../../ace-ig/ABIDE/ABIDE_I_2D/axial",
        #"../../ace-ig/ABIDE/ABIDE_I_2D/coronal",
        #"../../ace-ig/ABIDE/ABIDE_I_2D/sagittal",
        "../../ace-ig/ABIDE/ABIDE_II_2D/axial",
        #"../../ace-ig/ABIDE/ABIDE_II_2D/coronal",
        #"../../ace-ig/ABIDE/ABIDE_II_2D/sagittal"
    ]
    image_list = []
    age_list = []
    for folder_path in folder_paths:
        print(f"Processing folder: {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                subject_id = filename[2:7]
                dataset_num = 1 if "ABIDE_I_2D" in folder_path else 2
                age = get_ages(subject_id, dataset_num)
                if age is None:
                    continue
                age_list.append(age)
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert('L')
                img = img.resize((192, 192))
                img = np.array(img)
                img = normalize_slice(img)
                image_list.append(img)
    data = np.array(image_list)
    ages = np.array(age_list)
    return data, ages

def save_image_tensor(img_tensor, filename):
    # img_tensor shape: (1, H, W)
    img_np = img_tensor.squeeze().cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    img.save(filename)

# Training function per fold
def train_fold(train_loader, val_loader, encoder, decoder, generator, epochs, lr=1e-3): #changed epochs from 80 to 1
    encoder.train()
    decoder.train()
    generator.train()

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(generator.parameters()), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (imgs, ages) in enumerate(train_loader):
            imgs = imgs.to(device)
            ages = ages.to(device)

            optimizer.zero_grad()

            z_mean, z_log_var, z, r_mean, r_log_var, r = encoder(imgs)

            pz_mean, pz_log_var = generator(r)

            recon_imgs = decoder(z)

            loss = vae_loss_fn(imgs, recon_imgs, z_mean, z_log_var, pz_mean, pz_log_var, r_mean, r_log_var, ages)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return encoder, decoder, generator

# Evaluation function
def evaluate(val_loader, encoder):
    encoder.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for imgs, ages in val_loader:
            imgs = imgs.to(device)
            ages = ages.to(device)
            _, _, _, r_mean, _, _ = encoder(imgs)
            preds.append(r_mean.cpu().numpy())
            targets.append(ages.cpu().numpy())
    preds = np.concatenate(preds).squeeze()
    targets = np.concatenate(targets).squeeze()
    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)
    return preds, targets, mse, r2

def main():
    data, ages = load_data()
    print("Data shape:", data.shape)  # (N, H, W)
    print("Ages shape:", ages.shape)

    data = np.expand_dims(data, axis=1).astype(np.float32)  # (N, 1, H, W)
    ages = ages.astype(np.float32)
    #dataset = MRIDataset(data, ages, transform=train_transform)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #changed n_splits from 5 to 2 for testing
    fake_strat = np.zeros(len(ages))  
    preds_all = np.zeros(len(ages))

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, fake_strat)):
        print(f"Fold {fold+1}")

        train_dataset = MRIDataset(data[train_idx], ages[train_idx], transform=train_transform) #only augment training dataset
        val_dataset = MRIDataset(data[val_idx], ages[val_idx], transform=None)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        encoder = Encoder().to(device)
        decoder = Decoder().to(device)
        generator = Generator().to(device)

        encoder, decoder, generator = train_fold(train_loader, val_loader, encoder, decoder, generator, epochs=80) #changed epochs from 80 to 2 for testing

        preds, targets, mse, r2 = evaluate(val_loader, encoder)
        print(f"Validation MSE: {mse:.4f}, R2: {r2:.4f}")
        preds_all[val_idx] = preds

        # Save model weights 
        torch.save(encoder.state_dict(), f'torch_weights/augmented_axial_weights/encoder.pt')
        torch.save(decoder.state_dict(), f'torch_weights/augmented_axial_weights/decoder.pt')
        torch.save(generator.state_dict(), f'torch_weights/augmented_axial_weights/generator.pt')

    print("Cross-Validation MSE:", mean_squared_error(ages, preds_all))
    print("Cross-Validation R2:", r2_score(ages, preds_all))

    # Train on all data now to get final model
    print("Training on full dataset...")
    full_dataset = MRIDataset(data, ages, transform=train_transform)
    full_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    generator = Generator().to(device)

    encoder.train()
    decoder.train()
    generator.train()

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(generator.parameters()), lr=1e-3)

    for epoch in range(80): #changed epochs from 80 to 2 for testing
        total_loss = 0
        for imgs, ages_batch in full_loader:
            imgs = imgs.to(device)
            ages_batch = ages_batch.to(device)

            optimizer.zero_grad()

            z_mean, z_log_var, z, r_mean, r_log_var, r = encoder(imgs)

            pz_mean, pz_log_var = generator(r)

            recon_imgs = decoder(z)

            loss = vae_loss_fn(imgs, recon_imgs, z_mean, z_log_var, pz_mean, pz_log_var, r_mean, r_log_var, ages_batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Full data training epoch {epoch+1}/80 - Loss: {total_loss/len(full_loader):.4f}")

    # Save full models
    torch.save(encoder.state_dict(), 'torch_weights/augmented_axial_weights/encoder_final.pt')
    torch.save(decoder.state_dict(), 'torch_weights/augmented_axial_weights/decoder_final.pt')
    torch.save(generator.state_dict(), 'torch_weights/augmented_axial_weights/generator_final.pt')

    # Generate samples from latent space given age points
    generator.eval()
    decoder.eval()
    r_points = torch.tensor([-2, -1.5, -1, -0.5, 0, 1, 1.5, 2.5, 3.5, 4.5], dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        pz_mean, _ = generator(r_points)
        outputs = decoder(pz_mean)
    
    for i in range(outputs.shape[0]):
        img = outputs[i]
        #img = torch.sigmoid(img)  # normalize output between 0 and 1
        save_image_tensor(img, f'torch_generations_augmented/axial_generations/generated_{i}.png')

if __name__ == "__main__":
    main()
