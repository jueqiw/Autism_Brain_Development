import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import AutoEncoder


def reparameterize(mu, logvar):
    """Reparameterization trick for VAE"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VAEWithPretrainedAutoEncoder(nn.Module):
    """
    VAE that uses pretrained MONAI AutoEncoder as backbone
    Adds VAE latent space and age regression capabilities
    """

    def __init__(
        self,
        channels=(32, 64, 64),
        strides=(2, 2, 1),
        num_res_units=3,
        latent_dim=32,
        age_latent_dim=16,
        pretrained_path=None,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.age_latent_dim = age_latent_dim

        # Create the pretrained autoencoder
        self.pretrained_autoencoder = AutoEncoder(
            spatial_dims=2,
            in_channels=2,
            out_channels=1,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
        )

        # Load pretrained weights if provided
        if pretrained_path:
            print(f"Loading pretrained weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.pretrained_autoencoder.load_state_dict(state_dict)
            print("Pretrained weights loaded successfully")

        # Extract encoder and decoder from pretrained model
        self.encoder_backbone = self.pretrained_autoencoder.encode
        self.decoder_backbone = self.pretrained_autoencoder.decode

        # Get the bottleneck feature size from the pretrained model
        self._determine_feature_sizes()

        # VAE latent space layers
        self.fc_z_mean = nn.Linear(self.bottleneck_size, latent_dim)
        self.fc_z_logvar = nn.Linear(self.bottleneck_size, latent_dim)

        # Age regression layers
        self.fc_age_mean = nn.Linear(self.bottleneck_size, 1)
        self.fc_age_logvar = nn.Linear(self.bottleneck_size, 1)

        # Latent to bottleneck projection for decoder
        self.fc_latent_to_bottleneck = nn.Linear(latent_dim, self.bottleneck_size)

        # Age-dependent prior generator
        self.age_to_prior = AgePriorGenerator(latent_dim)

    def _determine_feature_sizes(self):
        """Determine the bottleneck feature size by running a dummy forward pass"""
        dummy_input = torch.randn(1, 2, 192, 192)

        with torch.no_grad():
            # Get encoder output (bottleneck features)
            bottleneck_features = self.encoder_backbone(dummy_input)

            # Flatten to get the size
            if isinstance(bottleneck_features, (list, tuple)):
                bottleneck_features = bottleneck_features[-1]

            # Get the flattened size
            self.bottleneck_size = bottleneck_features.view(
                bottleneck_features.size(0), -1
            ).size(1)
            self.bottleneck_shape = bottleneck_features.shape[1:]

        print(f"Bottleneck size determined: {self.bottleneck_size}")
        print(f"Bottleneck shape: {self.bottleneck_shape}")

    def encode(self, x):
        """Encode input to VAE latent space and age prediction"""
        # Get bottleneck features from pretrained encoder
        bottleneck_features = self.encoder_backbone(x)

        # Flatten bottleneck features
        if isinstance(bottleneck_features, (list, tuple)):
            bottleneck_features = bottleneck_features[-1]

        flat_features = bottleneck_features.view(bottleneck_features.size(0), -1)

        # VAE latent parameters
        z_mean = self.fc_z_mean(flat_features)
        z_logvar = self.fc_z_logvar(flat_features)
        z = reparameterize(z_mean, z_logvar)

        # Age regression parameters
        age_mean = self.fc_age_mean(flat_features)
        age_logvar = self.fc_age_logvar(flat_features)
        age_pred = reparameterize(age_mean, age_logvar)

        return z_mean, z_logvar, z, age_mean, age_logvar, age_pred

    def decode(self, z):
        """Decode from VAE latent space to output"""
        # Project latent back to bottleneck size
        bottleneck_flat = self.fc_latent_to_bottleneck(z)

        # Reshape to original bottleneck shape
        bottleneck_features = bottleneck_flat.view(-1, *self.bottleneck_shape)

        # Use pretrained decoder
        reconstruction = self.decoder_backbone(bottleneck_features)

        return reconstruction

    def forward(self, x, age=None):
        """Full forward pass"""
        # Encode
        z_mean, z_logvar, z, age_mean, age_logvar, age_pred = self.encode(x)

        # Decode
        reconstruction = self.decode(z)

        # Generate age-dependent prior if age is provided
        if age is not None:
            pz_mean, pz_logvar = self.age_to_prior(age)
            return (
                reconstruction,
                z_mean,
                z_logvar,
                pz_mean,
                pz_logvar,
                age_mean,
                age_logvar,
                age_pred,
            )
        else:
            return reconstruction, z_mean, z_logvar, age_mean, age_logvar, age_pred


class AgePriorGenerator(nn.Module):
    """Generate age-dependent prior for latent space"""

    def __init__(self, latent_dim=32, hidden_dim=128):
        super().__init__()

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.dropout = nn.Dropout(0.2)

    def forward(self, age):
        """Generate age-dependent prior parameters"""
        x = F.relu(self.fc1(age))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))

        pz_mean = self.fc_mean(x)
        pz_logvar = self.fc_logvar(x)

        return pz_mean, pz_logvar


def vae_loss_function(
    recon,
    target,
    input_img,
    z_mean,
    z_logvar,
    pz_mean,
    pz_logvar,
    age_mean,
    age_logvar,
    age_pred,
    true_age,
    brain_threshold=0.05,
    recon_weight=1.0,
    kl_weight=0.1,
    age_weight=1.0,
):
    """Combined VAE loss with brain masking and age regression"""

    # Create brain mask
    if input_img.dim() == 4:  # (batch, channels, H, W)
        brain_mask = (input_img.mean(dim=1, keepdim=True) > brain_threshold).float()
    else:
        brain_mask = (input_img > brain_threshold).float()

    # Reconstruction loss (only within brain)
    recon_loss = torch.abs(recon - target)

    # Apply brain mask and foreground weighting
    foreground_weight = 10.0
    target_weights = torch.where(torch.abs(target) > 0.1, foreground_weight, 1.0)
    final_weights = brain_mask * target_weights

    masked_recon_loss = final_weights * recon_loss

    brain_pixels = brain_mask.sum()
    if brain_pixels > 0:
        recon_loss = masked_recon_loss.sum() / brain_pixels
    else:
        recon_loss = masked_recon_loss.mean()

    # KL divergence between posterior and age-dependent prior
    kl_loss = (
        1
        + z_logvar
        - pz_logvar
        - ((z_mean - pz_mean).pow(2) / pz_logvar.exp())
        - (z_logvar.exp() / pz_logvar.exp())
    )
    kl_loss = -0.5 * kl_loss.sum(dim=1).mean()

    # Age regression loss
    age_loss = (
        0.5 * ((age_mean - true_age).pow(2) / age_logvar.exp()) + 0.5 * age_logvar
    )
    age_loss = age_loss.mean()

    # Combined loss
    total_loss = recon_weight * recon_loss + kl_weight * kl_loss + age_weight * age_loss

    return total_loss, recon_loss, kl_loss, age_loss


def create_vae_model(hparams, pretrained_path=None):
    """Factory function to create VAE model with pretrained weights"""

    # Parse model architecture parameters
    channels = tuple(map(int, hparams.model_channels.split(",")))
    strides = tuple(map(int, hparams.model_strides.split(",")))

    # Create VAE model
    model = VAEWithPretrainedAutoEncoder(
        channels=channels,
        strides=strides,
        num_res_units=hparams.num_res_units,
        latent_dim=getattr(hparams, "latent_dim", 32),
        pretrained_path=pretrained_path,
    )

    # Initialize new layers (VAE-specific ones)
    def init_new_layers(m):
        if isinstance(m, nn.Linear) and hasattr(m, "weight"):
            torch.nn.init.xavier_normal_(m.weight, gain=1.0)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    # Only initialize the new VAE layers, not pretrained ones
    model.fc_z_mean.apply(init_new_layers)
    model.fc_z_logvar.apply(init_new_layers)
    model.fc_age_mean.apply(init_new_layers)
    model.fc_age_logvar.apply(init_new_layers)
    model.fc_latent_to_bottleneck.apply(init_new_layers)
    model.age_to_prior.apply(init_new_layers)

    print("VAE model created with pretrained backbone")
    return model
