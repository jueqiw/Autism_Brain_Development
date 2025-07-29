import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import AutoEncoder
import torchvision.models as models


def reparameterize(mu, logvar):
    """Reparameterization trick for VAE"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using pretrained VGG features"""
    
    def __init__(self, feature_layers=None, use_normalization=True):
        super().__init__()
        if feature_layers is None:
            # Use multiple layers for richer perceptual comparison
            feature_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        
        self.feature_layers = feature_layers
        self.use_normalization = use_normalization
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True).features
        self.vgg = vgg
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.vgg.eval()
        
        # Layer name mapping for VGG16
        self.layer_name_mapping = {
            'relu1_2': 3,   # After ReLU of conv1_2
            'relu2_2': 8,   # After ReLU of conv2_2  
            'relu3_3': 15,  # After ReLU of conv3_3
            'relu4_3': 22,  # After ReLU of conv4_3
        }
        
    def normalize_batch(self, batch):
        """Normalize batch for VGG (ImageNet pretrained)"""
        if self.use_normalization:
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(batch.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(batch.device)
            return (batch - mean) / std
        return batch
    
    def get_features(self, x):
        """Extract features from specified layers"""
        # Convert single channel to 3 channels for VGG
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Normalize for VGG
        x = self.normalize_batch(x)
        
        features = {}
        for name, layer_idx in self.layer_name_mapping.items():
            if name in self.feature_layers:
                for i in range(layer_idx + 1):
                    x = self.vgg[i](x)
                features[name] = x.clone()
        
        return features
    
    def forward(self, pred, target):
        """Compute perceptual loss between predicted and target images"""
        pred_features = self.get_features(pred)
        target_features = self.get_features(target)
        
        perceptual_loss = 0
        for layer in self.feature_layers:
            # L2 loss between features
            loss = F.mse_loss(pred_features[layer], target_features[layer])
            perceptual_loss += loss
            
        return perceptual_loss / len(self.feature_layers)


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
        input_channels=3,  # 1 original + 2 conditional channels
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.age_latent_dim = age_latent_dim
        self.input_channels = input_channels

        self.channels = channels
        self.strides = strides
        self.num_res_units = num_res_units
        # Create autoencoder with 3 input channels from scratch
        self.base_autoencoder = AutoEncoder(
            spatial_dims=2,
            in_channels=input_channels,  # Use configurable input channels (3)
            out_channels=1,  # Still output 1 channel (reconstructed brain image)``
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
        )

        # Load pretrained weights for the encoder/decoder layers (except first conv)
        if pretrained_path:
            self._load_compatible_pretrained_weights(pretrained_path)

        # Extract encoder and decoder
        self.encoder_backbone = self.base_autoencoder.encode
        self.decoder_backbone = self.base_autoencoder.decode

        # Freeze all pretrained autoencoder parameters first
        for param in self.base_autoencoder.parameters():
            param.requires_grad = False

        # Define additional encoder/decoder layers BEFORE calculating feature sizes
        # Additional encoder layer to compress the latent space
        # This will reduce the large bottleneck size to something manageable for VAE
        self.additional_encoder = nn.Sequential(
            # First compression layer - reduce from 48x48 to 12x12
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 48x48 -> 24x24
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Second compression layer
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 24x24 -> 12x12
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Final compression layer - replace AdaptiveAvgPool2d with convolution
            nn.Conv2d(256, 128, kernel_size=3, stride=3, padding=0),  # 12x12 -> 4x4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Combined decoder layer to upsample back to pretrained decoder input size (48x48)
        self.combined_decoder = nn.Sequential(
            # First expansion layer - reverse of the compression (4x4 -> 12x12)
            nn.ConvTranspose2d(
                128, 256, kernel_size=3, stride=3, padding=0
            ),  # 4x4 -> 12x12
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Second expansion layer - from 12x12 to 24x24
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 12x12 -> 24x24
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Third expansion layer - from 24x24 to 48x48 (final size for decoder input)
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 24x24 -> 48x48
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Refinement layer to clean up features
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Keep 48x48
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.tanh = nn.Tanh()

        # NOW calculate feature sizes with the additional encoder
        self._determine_feature_sizes()
        print(f"Creating linear layers with bottleneck_size: {self.bottleneck_size}")
        # Create linear layers (apply tanh activation in forward pass)
        self.fc_z_mean = nn.Linear(self.bottleneck_size, latent_dim)
        self.fc_z_logvar = nn.Linear(self.bottleneck_size, latent_dim)

        self.fc_age_mean = nn.Linear(self.bottleneck_size, 1)
        self.fc_age_logvar = nn.Linear(self.bottleneck_size, 1)

        self.fc_latent_to_bottleneck = nn.Linear(latent_dim, self.bottleneck_size)

        # Age-dependent prior generator
        self.age_to_prior = AgePriorGenerator(latent_dim)
        
        # Perceptual loss network (frozen VGG features)
        self.perceptual_loss_fn = VGGPerceptualLoss(
            feature_layers=['relu2_2', 'relu3_3'],  # Use fewer layers for efficiency
            use_normalization=True
        )

    def _load_compatible_pretrained_weights(self, pretrained_path):
        """Load pretrained weights for compatible layers (skip first conv layer)"""

        pretrained_state_dict = torch.load(pretrained_path, map_location="cpu")

        # Get current model state dict
        current_state_dict = self.base_autoencoder.state_dict()

        compatible_weights = {}
        skipped_layers = []

        for key, value in pretrained_state_dict.items():
            if key in current_state_dict:
                current_shape = current_state_dict[key].shape
                pretrained_shape = value.shape

                # Skip first encoder layer due to channel mismatch
                if (
                    "encode_0" in key
                    and len(pretrained_shape) >= 2
                    and pretrained_shape[1] != current_shape[1]
                ):
                    skipped_layers.append(key)
                    continue

                # Load compatible layers
                if current_shape == pretrained_shape:
                    compatible_weights[key] = value
                else:
                    skipped_layers.append(key)
            else:
                skipped_layers.append(key)

        try:
            # Load compatible weights
            missing_keys, unexpected_keys = self.base_autoencoder.load_state_dict(
                compatible_weights, strict=False
            )

            print(f"Loaded {len(compatible_weights)} compatible pretrained weights")
            print(
                f"Skipped {len(skipped_layers)} incompatible layers (including first conv)"
            )
            print(
                f"First conv layer will use random initialization for 3-channel input"
            )

        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Using random initialization for all layers")

    def set_encoder_layer_trainable(self, layer_idx, trainable=True):
        """Set specific encoder layer as trainable or frozen"""
        layer_name = f"encode_{layer_idx}"
        if hasattr(self.encoder_backbone, layer_name):
            layer = getattr(self.encoder_backbone, layer_name)
            for param in layer.parameters():
                param.requires_grad = trainable

    def set_decoder_layer_trainable(self, layer_idx, trainable=True):
        """Set specific decoder layer as trainable or frozen"""
        if layer_idx == -1:
            # Set final decoder layer (last layer) as trainable
            decoder_layers = list(self.decoder_backbone.named_children())
            if decoder_layers:
                final_layer_name, final_layer = decoder_layers[-1]
                for param in final_layer.parameters():
                    param.requires_grad = trainable
                print(
                    f"Set final decoder layer '{final_layer_name}' trainable: {trainable}"
                )
        else:
            layer_name = f"decode_{layer_idx}"
            if hasattr(self.decoder_backbone, layer_name):
                layer = getattr(self.decoder_backbone, layer_name)
                for param in layer.parameters():
                    param.requires_grad = trainable

    def set_final_residual_block_trainable(self, trainable=True):
        """Set the final residual block (decode_2.resunit) as trainable with high learning rate"""
        if hasattr(self.decoder_backbone, "decode_2"):
            decode_2 = getattr(self.decoder_backbone, "decode_2")
            if hasattr(decode_2, "resunit"):
                resunit = getattr(decode_2, "resunit")
                for param in resunit.parameters():
                    param.requires_grad = trainable
                print(
                    f"Set final residual block (decode_2.resunit) trainable: {trainable}"
                )
            else:
                print("Warning: decode_2.resunit not found")

    def freeze_pretrained_encoder(self):
        """Freeze all pretrained encoder layers except the first conv"""
        for param in self.encoder_backbone.parameters():
            param.requires_grad = False
        # Keep first conv layer trainable
        self.set_encoder_layer_trainable(0, trainable=True)

    def freeze_pretrained_decoder(self):
        """Freeze all pretrained decoder layers except the final residual block"""
        for param in self.decoder_backbone.parameters():
            param.requires_grad = False
        # Keep final residual block trainable for better reconstruction
        self.set_final_residual_block_trainable(trainable=True)

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
        print("Creating parameter groups with different learning rates")

        # Group 1: First encoder layer (highest LR - randomly initialized for 3-channel input)
        first_encoder_params = []

        # Group 2: New VAE components (highest LR)
        new_component_params = []
        new_component_names = [
            "fc_z_mean",
            "fc_z_logvar",
            "fc_age_mean",
            "fc_age_logvar",
            "fc_latent_to_bottleneck",
            "additional_encoder",
            "combined_decoder",
            "age_to_prior",
        ]

        # Group 3: Final residual block (high LR)
        final_residual_params = []

        # Group 4: Other pretrained components (low LR)
        other_pretrained_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Check if it's the first encoder layer (needs highest LR)
            if "encoder_backbone.encode_0" in name:
                first_encoder_params.append(param)
            # Check if it's a new VAE component
            elif any(comp_name in name for comp_name in new_component_names):
                new_component_params.append(param)
            # Check if it's the final residual block (decode_2.resunit)
            elif "decoder_backbone.decode_2.resunit" in name:
                final_residual_params.append(param)
            # Check if it's any other decoder backbone component
            elif "decoder_backbone" in name:
                other_pretrained_params.append(param)
            # Everything else that's trainable
            else:
                other_pretrained_params.append(param)

        param_groups = []

        if first_encoder_params:
            param_groups.append(
                {
                    "params": first_encoder_params,
                    "lr": new_lr * 1.2,  # Highest LR for conditional input learning
                    "weight_decay": 0.0,
                    "name": "first_encoder",
                }
            )
            print(
                f"First encoder layer: {len(first_encoder_params)} parameters with LR {new_lr * 1.2}"
            )

        if new_component_params:
            param_groups.append(
                {
                    "params": new_component_params,
                    "lr": new_lr,
                    "weight_decay": 0.0,
                    "name": "new_components",
                }
            )
            print(
                f"New components: {len(new_component_params)} parameters with LR {new_lr}"
            )

        if final_residual_params:
            param_groups.append(
                {
                    "params": final_residual_params,
                    "lr": new_lr * 1.0,  # High LR for final residual block
                    "weight_decay": weight_decay * 0.1,
                    "name": "final_residual_block",
                }
            )
            print(
                f"Final residual block: {len(final_residual_params)} parameters with LR {new_lr * 1.0}"
            )

        if other_pretrained_params:
            param_groups.append(
                {
                    "params": other_pretrained_params,
                    "lr": pretrained_lr,
                    "weight_decay": weight_decay,
                    "name": "pretrained",
                }
            )
            print(
                f"Other pretrained: {len(other_pretrained_params)} parameters with LR {pretrained_lr}"
            )

        return param_groups

    def _determine_feature_sizes(self):
        """Determine the bottleneck feature size by running a dummy forward pass"""
        dummy_input = torch.randn(1, self.input_channels, 192, 192)

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
        z_mean = self.tanh(self.fc_z_mean(flat_features))  # Apply tanh to z_mean
        z_logvar = self.fc_z_logvar(flat_features)  # No tanh for logvar
        z = reparameterize(z_mean, z_logvar)

        # Age regression parameters
        age_mean = self.tanh(self.fc_age_mean(flat_features))  # Apply tanh to age_mean
        age_logvar = self.fc_age_logvar(flat_features)  # No tanh for logvar
        age_pred = reparameterize(age_mean, age_logvar)

        return z_mean, z_logvar, z, age_mean, age_logvar, age_pred

    def decode(self, z):
        """Decode from VAE latent space to output"""
        # Project latent back to compressed bottleneck size
        compressed_flat = self.fc_latent_to_bottleneck(z)

        # Reshape to compressed bottleneck shape
        compressed_features = compressed_flat.view(-1, *self.bottleneck_shape)

        # Pass through combined decoder layers to expand back to decoder input size
        decoder_input = self.combined_decoder(compressed_features)

        # Use pretrained decoder (outputs 192x192 directly)
        # The final residual block (decode_2.resunit) will have high learning rate
        reconstruction = self.decoder_backbone(decoder_input)

        return reconstruction

    def forward(self, x, age=None):
        # Encode
        z_mean, z_logvar, z, age_mean, age_logvar, age_pred = self.encode(x)

        # Decode
        reconstruction = self.tanh(self.decode(z))

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
    perceptual_weight=0.1,
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

    # Perceptual loss using VGG features (create instance if needed)
    perceptual_loss = 0.0
    if perceptual_weight > 0:
        # Create VGG perceptual loss on the fly
        vgg_loss = VGGPerceptualLoss(
            feature_layers=['relu2_2', 'relu3_3'], 
            use_normalization=True
        ).to(recon.device)
        perceptual_loss = vgg_loss(recon, target)
    
    # Combined loss
    total_loss = (recon_weight * recon_loss + 
                 kl_weight * kl_loss + 
                 age_weight * age_loss + 
                 perceptual_weight * perceptual_loss)

    return total_loss, recon_loss, kl_loss, age_loss, perceptual_loss


def create_vae_model(hparams, pretrained_path=None):
    """Factory function to create VAE model with pretrained weights"""

    # Create VAE model
    model = VAEWithPretrainedAutoEncoder(
        channels=(32, 64, 64),
        strides=(2, 2, 1),
        num_res_units=3,
        latent_dim=getattr(hparams, "latent_dim", 32),
        input_channels=getattr(hparams, "input_channels", 3),
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


def save_vae_model(
    model, filepath, hparams=None, epoch=None, optimizer=None, loss_history=None
):
    """
    Save VAE model with all necessary components for reconstruction

    Args:
        model: VAEWithPretrainedAutoEncoder model
        filepath: Path to save the model
        hparams: Hyperparameters used to create the model
        epoch: Current epoch (optional)
        optimizer: Optimizer state (optional)
        loss_history: Training loss history (optional)
    """
    save_dict = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "channels": model.channels,
            "strides": model.strides,
            "num_res_units": model.num_res_units,
            "latent_dim": model.latent_dim,
            "age_latent_dim": model.age_latent_dim,
            "input_channels": model.input_channels,
            "bottleneck_size": model.bottleneck_size,
            "bottleneck_shape": model.bottleneck_shape,
        },
    }

    torch.save(save_dict, filepath)
    print(f"VAE model saved to {filepath}")

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model summary: {total_params:,} total parameters, {trainable_params:,} trainable"
    )


def load_vae_model(filepath, device="cpu", load_optimizer=False, strict=True):
    """
    Load VAE model from saved checkpoint

    Args:
        filepath: Path to the saved model
        device: Device to load the model on
        load_optimizer: Whether to return optimizer state (if available)
        strict: Whether to strictly enforce state dict loading

    Returns:
        model: Loaded VAE model
        checkpoint_info: Dictionary with additional loaded information
    """
    checkpoint = torch.load(filepath, map_location=device)

    # Check if this is an old-style save (just state_dict) or new-style save (with model_config)
    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        # New-style save with model_config
        model_config = checkpoint["model_config"]
    else:
        # Old-style save - just state_dict, use default configuration
        print("Warning: Loading old-style model save. Using default configuration.")
        model_config = {
            "channels": (32, 64, 64),
            "strides": (2, 2, 1),
            "num_res_units": 3,
            "latent_dim": 32,
            "age_latent_dim": 16,
            "input_channels": 3,
            "bottleneck_size": 2048,  # Default value
            "bottleneck_shape": (128, 4, 4),  # Default value
        }
        # If checkpoint is just the state_dict, wrap it
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            checkpoint = {"model_state_dict": checkpoint}

    # Recreate the model
    model = VAEWithPretrainedAutoEncoder(
        channels=model_config["channels"],
        strides=model_config["strides"],
        num_res_units=model_config["num_res_units"],
        latent_dim=model_config["latent_dim"],
        age_latent_dim=model_config.get("age_latent_dim", 16),
        input_channels=model_config["input_channels"],
        pretrained_path=None,  # Don't reload pretrained weights when loading saved model
    )

    # Load the saved state
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state_dict"], strict=strict
    )

    if missing_keys:
        print(f"Warning: Missing keys when loading model: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys when loading model: {unexpected_keys}")

    model.to(device)

    # Prepare checkpoint info
    checkpoint_info = {
        "model_config": model_config,
        "epoch": checkpoint.get("epoch", None),
        "hparams": checkpoint.get("hparams", None),
        "loss_history": checkpoint.get("loss_history", None),
        "save_timestamp": checkpoint.get("save_timestamp", None),
    }

    # Handle optimizer state if requested
    if load_optimizer and "optimizer_state_dict" in checkpoint:
        checkpoint_info["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]

    print(f"VAE model loaded from {filepath}")
    if checkpoint_info["epoch"] is not None:
        print(f"Model was saved at epoch {checkpoint_info['epoch']}")
    if checkpoint_info["save_timestamp"]:
        print(f"Model was saved on {checkpoint_info['save_timestamp']}")

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Loaded model: {total_params:,} total parameters, {trainable_params:,} trainable"
    )

    return model, checkpoint_info
