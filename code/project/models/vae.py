import sys

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

        self.channels = channels
        self.strides = strides
        self.num_res_units = num_res_units
        self.base_autoencoder = AutoEncoder(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
        )

        if pretrained_path:
            self._load_and_adapt_pretrained_weights(pretrained_path)

        # Extract encoder and decoder
        self.encoder_backbone = self.base_autoencoder.encode
        self.decoder_backbone = self.base_autoencoder.decode

        # Freeze all pretrained autoencoder parameters first
        for param in self.base_autoencoder.parameters():
            param.requires_grad = False

        self._determine_feature_sizes()

        print(f"Creating linear layers with bottleneck_size: {self.bottleneck_size}")
        self.fc_z_mean = nn.Linear(self.bottleneck_size, latent_dim)
        self.fc_z_logvar = nn.Linear(self.bottleneck_size, latent_dim)

        self.fc_age_mean = nn.Linear(self.bottleneck_size, 1)
        self.fc_age_logvar = nn.Linear(self.bottleneck_size, 1)

        self.fc_latent_to_bottleneck = nn.Linear(latent_dim, self.bottleneck_size)

        # Additional encoder layer to compress the latent space
        # This will reduce the large bottleneck size to something manageable for VAE
        self.additional_encoder = nn.Sequential(
            # Aggressive pooling to reduce 48x48 to 2x2 for small latent space
            nn.AdaptiveAvgPool2d((2, 2)),  # 64 * 2 * 2 = 256
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Corresponding decoder layer to upsample back
        self.additional_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Final interpolation layer to match pretrained decoder input size
        self.interpolate_to_decoder = nn.Upsample(size=(48, 48), mode='bilinear', align_corners=False)

        # Age-dependent prior generator
        self.age_to_prior = AgePriorGenerator(latent_dim)

    def _load_and_adapt_pretrained_weights(self, pretrained_path):
        """Load pretrained weights and adapt first conv layer from 2-channel to 1-channel"""
        print(f"Loading pretrained weights from {pretrained_path}")

        # Load the pretrained state dict
        pretrained_state_dict = torch.load(pretrained_path, map_location="cpu")

        # Create a new state dict for the 1-channel model
        adapted_state_dict = {}

        # Get the 2-channel first conv layer weights
        first_conv_key = "encode.encode_0.conv.unit0.conv.weight"
        if first_conv_key in pretrained_state_dict:
            original_weights = pretrained_state_dict[first_conv_key]
            print(f"Original first conv weight shape: {original_weights.shape}")

            adapted_weights = original_weights.mean(dim=1, keepdim=True)
            print(f"Adapted first conv weight shape: {adapted_weights.shape}")

            adapted_state_dict[first_conv_key] = adapted_weights

        for key, value in pretrained_state_dict.items():
            if key != first_conv_key:
                adapted_state_dict[key] = value

        try:
            missing_keys, unexpected_keys = self.base_autoencoder.load_state_dict(
                adapted_state_dict, strict=False
            )
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
            print("Weights adapted and loaded successfully")
            
            # Verify the first conv layer has correct shape
            first_conv_weight = self.base_autoencoder.encode.encode_0.conv.unit0.conv.weight
            print(f"Loaded first conv weight shape: {first_conv_weight.shape}")
            
        except Exception as e:
            print(f"Error loading adapted weights: {e}")
            print("Falling back to random initialization")

    def set_encoder_layer_trainable(self, layer_idx, trainable=True):
        """Set specific encoder layer as trainable or frozen"""
        layer_name = f"encode_{layer_idx}"
        if hasattr(self.encoder_backbone, layer_name):
            layer = getattr(self.encoder_backbone, layer_name)
            for param in layer.parameters():
                param.requires_grad = trainable

    def set_decoder_layer_trainable(self, layer_idx, trainable=True):
        """Set specific decoder layer as trainable or frozen"""
        layer_name = f"decode_{layer_idx}"
        if hasattr(self.decoder_backbone, layer_name):
            layer = getattr(self.decoder_backbone, layer_name)
            for param in layer.parameters():
                param.requires_grad = trainable

    def freeze_pretrained_encoder(self):
        """Freeze all pretrained encoder layers except the first conv"""
        for param in self.encoder_backbone.parameters():
            param.requires_grad = False
        # Keep first conv layer trainable
        self.set_encoder_layer_trainable(0, trainable=True)

    def freeze_pretrained_decoder(self):
        """Freeze all pretrained decoder layers"""
        for param in self.decoder_backbone.parameters():
            param.requires_grad = False

    def unfreeze_encoder_layer_by_layer(self, stage):
        """Gradually unfreeze encoder layers from deeper to shallower"""
        # Stage 0: only first conv + VAE heads (warm-up)
        # Stage 1: unfreeze deepest layers first
        # Stage 2: unfreeze middle layers
        # Stage 3: unfreeze all layers

        if stage >= 1:
            # Unfreeze deeper layers first (encode_2, encode_1, etc.)
            for layer_idx in reversed(range(1, 3)):  # encode_2, encode_1
                if stage >= (3 - layer_idx):
                    self.set_encoder_layer_trainable(layer_idx, trainable=True)

    def get_parameter_groups(self, new_lr=1e-3, pretrained_lr=1e-4, weight_decay=1e-4):
        """Get parameter groups with different learning rates and regularization"""
        new_params = []
        pretrained_params = []

        # New VAE-specific layers (higher LR, no weight decay)
        for module in [
            self.fc_z_mean,
            self.fc_z_logvar,
            self.fc_age_mean,
            self.fc_age_logvar,
            self.fc_latent_to_bottleneck,
            self.age_to_prior,
            self.additional_encoder,
            self.additional_decoder,
        ]:
            try:
                module_params = [p for p in module.parameters() if p.requires_grad]
                print(
                    f"Adding {len(module_params)} params from {module.__class__.__name__}"
                )
                for i, p in enumerate(module_params):
                    print(f"  Param {i}: shape {p.shape}")
                new_params.extend(module_params)
            except Exception as e:
                print(f"Error processing module {module.__class__.__name__}: {e}")
                print(f"Module: {module}")
                raise

        # First conv layer from encoder (higher LR, no weight decay)
        if hasattr(self.encoder_backbone, "encode_0"):
            new_params.extend(
                [
                    p
                    for p in self.encoder_backbone.encode_0.parameters()
                    if p.requires_grad
                ]
            )

        # Pretrained layers (lower LR, with weight decay for regularization)
        for module in [self.encoder_backbone, self.decoder_backbone]:
            for name, param in module.named_parameters():
                if param.requires_grad and param not in new_params:
                    pretrained_params.append(param)

        return [
            {"params": new_params, "lr": new_lr, "weight_decay": 0.0},
            {
                "params": pretrained_params,
                "lr": pretrained_lr,
                "weight_decay": weight_decay,
            },
        ]

    def _determine_feature_sizes(self):
        """Determine the bottleneck feature size by running a dummy forward pass"""
        dummy_input = torch.randn(1, 1, 192, 192)

        with torch.no_grad():
            # Get encoder output (bottleneck features from pretrained encoder)
            bottleneck_features = self.encoder_backbone(dummy_input)

            # Flatten to get the size
            if isinstance(bottleneck_features, (list, tuple)):
                bottleneck_features = bottleneck_features[-1]

            print(f"Pretrained encoder output shape: {bottleneck_features.shape}")
            
            # Pass through additional encoder layers to compress further
            compressed_features = self.additional_encoder(bottleneck_features)
            print(f"After additional encoder shape: {compressed_features.shape}")
            
            # Flatten to get the compressed size for VAE
            self.bottleneck_size = compressed_features.view(
                compressed_features.size(0), -1
            ).size(1)
            self.bottleneck_shape = compressed_features.shape[1:]

        print(f"Final bottleneck size for VAE: {self.bottleneck_size}")
        print(f"Final bottleneck shape: {self.bottleneck_shape}")

    def encode(self, x):
        """Encode input to VAE latent space and age prediction"""
        # Get bottleneck features from pretrained encoder
        bottleneck_features = self.encoder_backbone(x)

        # Flatten bottleneck features
        if isinstance(bottleneck_features, (list, tuple)):
            bottleneck_features = bottleneck_features[-1]

        # Pass through additional encoder layers to compress further
        compressed_features = self.additional_encoder(bottleneck_features)
        
        # Flatten compressed features
        flat_features = compressed_features.view(compressed_features.size(0), -1)

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
        # Project latent back to compressed bottleneck size
        compressed_flat = self.fc_latent_to_bottleneck(z)

        # Reshape to compressed bottleneck shape
        compressed_features = compressed_flat.view(-1, *self.bottleneck_shape)

        # Pass through additional decoder layers to expand back
        expanded_features = self.additional_decoder(compressed_features)
        
        # Interpolate to match pretrained decoder input size (48x48)
        decoder_input = self.interpolate_to_decoder(expanded_features)

        # Use pretrained decoder
        reconstruction = self.decoder_backbone(decoder_input)

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

    # Create VAE model
    model = VAEWithPretrainedAutoEncoder(
        channels=(32, 64, 64),
        strides=(2, 2, 1),
        num_res_units=3,
        latent_dim=getattr(hparams, "latent_dim", 32),
        pretrained_path=pretrained_path,
    )

    def init_new_layers(m):
        if isinstance(m, nn.Linear) and hasattr(m, "weight"):
            torch.nn.init.xavier_normal_(m.weight, gain=1.0)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.fc_z_mean.apply(init_new_layers)
    model.fc_z_logvar.apply(init_new_layers)
    model.fc_age_mean.apply(init_new_layers)
    model.fc_age_logvar.apply(init_new_layers)
    model.fc_latent_to_bottleneck.apply(init_new_layers)
    model.age_to_prior.apply(init_new_layers)

    print("VAE model created with pretrained backbone")
    return model
