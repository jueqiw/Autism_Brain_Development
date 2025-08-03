import torch
import torch.nn as nn
import torch.nn.functional as F

# Model version to ensure we're using the updated architecture
MODEL_VERSION = "v2.1_fixed_channels"


class ConvBNReLU(nn.Module):
    """Conv2D + BatchNorm + ReLU (matching conv2D_layer_bn from original)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # He normal initialization (matching kernel_initializer="he_normal")
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UpsampleConvBNReLU(nn.Module):
    """Upsample + Conv2D + BatchNorm + ReLU (replacing transposed convolution for better quality)"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,  # Kept for compatibility but not used
    ):
        super().__init__()
        # Use bilinear upsampling for smoother results
        self.upsample = nn.Upsample(
            scale_factor=stride, mode="bilinear", align_corners=False
        )
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # He normal initialization
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class GUNet16_2D_BN_with_AD(nn.Module):
    """
    Exact PyTorch conversion of G_unet_16_2D_bn_with_AD from GAN.py
    Line-by-line conversion from Keras to PyTorch
    """

    def __init__(
        self,
        input_shape=(1, 196, 196),
        filters=32,
        latent_space=128,
        age_dim=20,
        AD_dim=1,
        activation="tanh",
    ):
        super().__init__()

        self.input_shape = input_shape
        self.filters = filters  # f in original code
        self.latent_space = latent_space
        self.age_dim = age_dim
        self.AD_dim = AD_dim

        f = filters

        # ========== ENCODER PATH (exact match to original) ==========
        # Line 20-29: conv1_1
        self.conv1_1 = ConvBNReLU(input_shape[0], f, kernel_size=3, stride=1, padding=1)

        # Line 30-39: conv1_2
        self.conv1_2 = ConvBNReLU(f, f, kernel_size=3, stride=1, padding=1)

        # Line 40: pool1 = MaxPool2D()(conv1_2)
        self.pool1 = nn.MaxPool2d(2)

        # Line 43-52: conv2_1
        self.conv2_1 = ConvBNReLU(f, f * 2, kernel_size=3, stride=1, padding=1)

        # Line 53-62: conv2_2
        self.conv2_2 = ConvBNReLU(f * 2, f * 2, kernel_size=3, stride=1, padding=1)

        # Line 63: pool2 = MaxPool2D()(conv2_2)
        self.pool2 = nn.MaxPool2d(2)

        # Line 65-74: conv3_1
        self.conv3_1 = ConvBNReLU(f * 2, f * 4, kernel_size=3, stride=1, padding=1)

        # Line 75-84: conv3_2
        self.conv3_2 = ConvBNReLU(f * 4, f * 4, kernel_size=3, stride=1, padding=1)

        # Line 85: pool3 = MaxPool2D()(conv3_2)
        self.pool3 = nn.MaxPool2d(2)

        # Line 86-95: conv4_1
        self.conv4_1 = ConvBNReLU(f * 4, f * 8, kernel_size=3, stride=1, padding=1)

        # Line 96-105: conv4_2
        self.conv4_2 = ConvBNReLU(f * 8, f * 8, kernel_size=3, stride=1, padding=1)

        # Line 106: pool4 = MaxPool2D()(conv4_2)
        self.pool4 = nn.MaxPool2d(2)

        # ========== ADDITIONAL LAYER FOR DEEPER ARCHITECTURE ==========
        # conv5_1 and conv5_2 for deeper representation
        self.conv5_1 = ConvBNReLU(f * 8, f * 8, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = ConvBNReLU(f * 8, f * 8, kernel_size=3, stride=1, padding=1)

        # pool5 for final downsampling
        self.pool5 = nn.MaxPool2d(2)

        # ========== BOTTLENECK WITH CONDITIONING (adjusted for deeper architecture) ==========
        # Connect to conv5 output (f * 8 channels)
        self.mid1_1 = ConvBNReLU(f * 8, f, kernel_size=3, stride=1, padding=1)

        # Calculate flattened size (matching comment: batch size, 4160)
        self.flatten_size = self._calculate_flatten_size()

        # Line 122-124: dens1_1 = Dense(units=self.conf.latent_space, activation="sigmoid")
        self.dens1_1 = nn.Linear(self.flatten_size, latent_space)
        # Line 125: dens1_1 = BatchNormalization()(dens1_1)
        self.bn1_1 = nn.BatchNorm1d(latent_space)

        # Line 132-134: dens2_1 = Dense(units=self.conf.latent_space, activation="relu")
        self.dens2_1 = nn.Linear(latent_space + age_dim, latent_space)
        # Line 135: dens2_1 = BatchNormalization()(dens2_1)
        self.bn2_1 = nn.BatchNorm1d(latent_space)

        # Line 139: dens3_1 = Dense(units=4160, activation="relu")
        self.dens3_1 = nn.Linear(latent_space + AD_dim, self.flatten_size)

        # ========== DECODER PATH (adjusted for deeper architecture) ==========
        # New upconv5 for the deepest layer
        self.upconv5 = UpsampleConvBNReLU(
            f * 8 + f, f * 8, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        # conv5 processing layers
        self.conv_m_0_1 = ConvBNReLU(
            f * 8 + f * 8, f * 8, kernel_size=3, stride=1, padding=1
        )
        self.conv_m_0_2 = ConvBNReLU(f * 8, f * 8, kernel_size=3, stride=1, padding=1)

        # Line 145-154: upconv4
        self.upconv4 = UpsampleConvBNReLU(
            f * 8, f * 8, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        # Line 157-166: conv_m_1_1
        self.conv_m_1_1 = ConvBNReLU(
            f * 8 + f * 8, f * 8, kernel_size=3, stride=1, padding=1
        )

        # Line 167-176: conv_m_1_2
        self.conv_m_1_2 = ConvBNReLU(f * 8, f * 8, kernel_size=3, stride=1, padding=1)

        # Line 179-188: upconv3
        self.upconv3 = UpsampleConvBNReLU(
            f * 8, f * 4, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        # Line 194-203: convD_1 (renamed to avoid conflict with encoder conv5_1)
        self.convD_1 = ConvBNReLU(
            f * 4 + f * 4, f * 4, kernel_size=3, stride=1, padding=1
        )

        # Line 204-213: convD_2 (renamed to avoid conflict with encoder conv5_2)
        self.convD_2 = ConvBNReLU(f * 4, f * 4, kernel_size=3, stride=1, padding=1)

        # Line 216-225: upconv2
        self.upconv2 = UpsampleConvBNReLU(
            f * 4, f * 2, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        # Line 231-240: conv6_1
        self.conv6_1 = ConvBNReLU(
            f * 2 + f * 2, f * 2, kernel_size=3, stride=1, padding=1
        )

        # Line 241-250: conv6_2
        self.conv6_2 = ConvBNReLU(f * 2, f * 2, kernel_size=3, stride=1, padding=1)

        # Line 252-261: upconv1
        self.upconv1 = UpsampleConvBNReLU(
            f * 2, f, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        # Line 267-276: conv8_1
        self.conv8_1 = ConvBNReLU(f + f, f, kernel_size=3, stride=1, padding=1)

        # Line 277-286: conv8_2 (final output layer)
        self.conv8_2 = nn.Conv2d(f, 1, kernel_size=3, stride=1, padding=1)

        # Set final activation (line 284: activation=self.conf.G_activation)
        if activation == "tanh":
            self.final_activation = nn.Tanh()
        elif activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif activation == "linear":
            self.final_activation = nn.Identity()

        # Initialize final layer
        nn.init.kaiming_normal_(
            self.conv8_2.weight, mode="fan_out", nonlinearity="relu"
        )
        if self.conv8_2.bias is not None:
            nn.init.zeros_(self.conv8_2.bias)

    def _calculate_flatten_size(self):
        """Calculate flattened size after 5 pooling operations"""
        with torch.no_grad():
            x = torch.zeros(1, *self.input_shape)
            # Apply 5 pooling operations for deeper architecture
            for _ in range(5):
                x = F.max_pool2d(x, 2)
            # Apply mid1_1 conv to get final spatial size
            x = torch.zeros(1, self.filters, x.shape[2], x.shape[3])
            return x.view(1, -1).shape[1]

    def forward(self, g_input, age_vector, AD_vector):
        """
        Forward pass - exact conversion from original Keras model

        Args:
            g_input: Input image tensor (line 18: g_input = Input(shape=self.conf.input_shape))
            age_vector: Age vector (line 126: age_vector = Input(shape=(self.conf.age_dim,)))
            AD_vector: AD vector (line 136: AD_vector = Input(shape=(self.conf.AD_dim,)))
        """

        # ========== ENCODER PATH ==========
        # Line 20-40: Encoder layers with pooling
        conv1_1 = self.conv1_1(g_input)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)  # (batch size, 80, 104, filters)

        # Line 43-63: Second encoder block
        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)

        # Line 65-85: Third encoder block
        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        pool3 = self.pool3(conv3_2)

        # Line 86-106: Fourth encoder block
        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        pool4 = self.pool4(conv4_2)

        # ========== NEW FIFTH ENCODER BLOCK ==========
        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        pool5 = self.pool5(conv5_2)

        # ========== BOTTLENECK WITH CONDITIONING ==========
        # Line 109-118: Middle layer - connect to deepest pool5
        mid1_1 = self.mid1_1(pool5)  # (batch size, smaller spatial size, filter)

        # Line 120: flat1_1 = Flatten()(mid1_1)
        flat1_1 = mid1_1.view(mid1_1.size(0), -1)  # (batch size, 4160)

        # Line 122-125: dens1_1 = Dense(..., activation="sigmoid") + BatchNormalization
        dens1_1 = torch.sigmoid(self.bn1_1(self.dens1_1(flat1_1)))

        # Line 129: mid_concat1_1 = Concatenate()([dens1_1, age_vector])
        # Ensure age_vector is on the same device as dens1_1
        age_vector = age_vector.to(dens1_1.device)
        mid_concat1_1 = torch.cat(
            [dens1_1, age_vector], dim=1
        )  # (batch_size, 130+Age_dim)

        # Line 132-135: dens2_1 = Dense(..., activation="relu") + BatchNormalization
        dens2_1 = F.relu(self.bn2_1(self.dens2_1(mid_concat1_1)))

        # Line 137: mid_concat2_1 = Concatenate()([dens2_1, AD_vector])
        # Ensure AD_vector is on the same device as dens2_1
        AD_vector = AD_vector.to(dens2_1.device)
        mid_concat2_1 = torch.cat(
            [dens2_1, AD_vector], dim=1
        )  # (batch size, 130+AD_dim)

        # Line 139: dens3_1 = Dense(units=4160, activation="relu")
        dens3_1 = F.relu(self.dens3_1(mid_concat2_1))  # (batch size, 4160)

        # Line 141: rshape1_1 = Reshape to match pool5 spatial dimensions
        batch_size = dens3_1.size(0)
        spatial_h, spatial_w = pool5.shape[2], pool5.shape[3]
        rshape1_1 = dens3_1.view(
            batch_size, self.filters, spatial_h, spatial_w
        )  # (batch size, smaller spatial size, 32)

        # Line 143: mid_concat3_1 = Concatenate()([pool5, rshape1_1])
        mid_concat3_1 = torch.cat(
            [pool5, rshape1_1], dim=1
        )  # (batch size, smaller spatial size, 32+f*8)

        # ========== DECODER PATH WITH SKIP CONNECTIONS ==========
        # NEW: upconv5 for deepest layer
        upconv5 = self.upconv5(mid_concat3_1)
        # Handle size mismatch with padding/cropping for conv5_2
        target_h, target_w = conv5_2.shape[2], conv5_2.shape[3]
        if upconv5.shape[2] != target_h or upconv5.shape[3] != target_w:
            if upconv5.shape[2] < target_h or upconv5.shape[3] < target_w:
                pad_h = max(0, target_h - upconv5.shape[2])
                pad_w = max(0, target_w - upconv5.shape[3])
                upconv5 = F.pad(upconv5, (0, pad_w, 0, pad_h))
            else:
                upconv5 = upconv5[:, :, :target_h, :target_w]
        concat5 = torch.cat([upconv5, conv5_2], dim=1)

        # Process the concat5 features
        conv_m_0_1 = self.conv_m_0_1(concat5)
        conv_m_0_2 = self.conv_m_0_2(conv_m_0_1)

        # Line 145-155: upconv4 + concat4
        upconv4 = self.upconv4(conv_m_0_2)
        # Handle size mismatch with padding/cropping
        target_h, target_w = conv4_2.shape[2], conv4_2.shape[3]
        if upconv4.shape[2] != target_h or upconv4.shape[3] != target_w:
            if upconv4.shape[2] < target_h or upconv4.shape[3] < target_w:
                pad_h = max(0, target_h - upconv4.shape[2])
                pad_w = max(0, target_w - upconv4.shape[3])
                upconv4 = F.pad(upconv4, (0, pad_w, 0, pad_h))
            else:
                upconv4 = upconv4[:, :, :target_h, :target_w]
        concat4 = torch.cat([upconv4, conv4_2], dim=1)

        # Line 157-176: conv_m_1_1 and conv_m_1_2
        conv_m_1_1 = self.conv_m_1_1(concat4)
        conv_m_1_2 = self.conv_m_1_2(conv_m_1_1)

        # Line 179-191: upconv3 + concat3
        upconv3 = self.upconv3(conv_m_1_2)
        target_h, target_w = conv3_2.shape[2], conv3_2.shape[3]
        if upconv3.shape[2] != target_h or upconv3.shape[3] != target_w:
            if upconv3.shape[2] < target_h or upconv3.shape[3] < target_w:
                pad_h = max(0, target_h - upconv3.shape[2])
                pad_w = max(0, target_w - upconv3.shape[3])
                upconv3 = F.pad(upconv3, (0, pad_w, 0, pad_h))
            else:
                upconv3 = upconv3[:, :, :target_h, :target_w]
        concat3 = torch.cat([upconv3, conv3_2], dim=1)

        # Line 194-213: convD_1 and convD_2
        convD_1 = self.convD_1(concat3)
        convD_2 = self.convD_2(convD_1)

        # Line 216-228: upconv2 + concat2
        upconv2 = self.upconv2(convD_2)
        target_h, target_w = conv2_2.shape[2], conv2_2.shape[3]
        if upconv2.shape[2] != target_h or upconv2.shape[3] != target_w:
            if upconv2.shape[2] < target_h or upconv2.shape[3] < target_w:
                pad_h = max(0, target_h - upconv2.shape[2])
                pad_w = max(0, target_w - upconv2.shape[3])
                upconv2 = F.pad(upconv2, (0, pad_w, 0, pad_h))
            else:
                upconv2 = upconv2[:, :, :target_h, :target_w]
        concat2 = torch.cat([upconv2, conv2_2], dim=1)

        # Line 231-250: conv6_1 and conv6_2
        conv6_1 = self.conv6_1(concat2)
        conv6_2 = self.conv6_2(conv6_1)

        # Line 252-264: upconv1 + concat1
        upconv1 = self.upconv1(conv6_2)
        target_h, target_w = conv1_2.shape[2], conv1_2.shape[3]
        if upconv1.shape[2] != target_h or upconv1.shape[3] != target_w:
            if upconv1.shape[2] < target_h or upconv1.shape[3] < target_w:
                pad_h = max(0, target_h - upconv1.shape[2])
                pad_w = max(0, target_w - upconv1.shape[3])
                upconv1 = F.pad(upconv1, (0, pad_w, 0, pad_h))
            else:
                upconv1 = upconv1[:, :, :target_h, :target_w]
        concat1 = torch.cat(
            [upconv1, conv1_2], dim=1
        )  # (batch size, 160, 208, filters*2)

        # Line 267-276: conv8_1
        conv8_1 = self.conv8_1(concat1)

        # Line 277-286: conv8_2 (final output)
        conv8_2 = self.conv8_2(conv8_1)  # (batch size, 160, 208, 1)

        # Apply final activation
        residual = self.final_activation(conv8_2)

        # Add residual to input image to get final generated image
        output = g_input + residual

        return output


def create_gan_generator(
    input_shape=(1, 196, 196),  # Single channel input (no conditional channels)
    filters=32,
    latent_space=128,
    age_dim=20,
    AD_dim=1,
    activation="tanh",
):
    """
    Factory function to create GAN generator - exact match to original build() method

    Args:
        input_shape: Input image shape (channels, height, width)
        filters: Base number of filters (f in original)
        latent_space: Latent space dimension (self.conf.latent_space in original)
        age_dim: Age vector dimension (self.conf.age_dim in original)
        AD_dim: AD vector dimension (self.conf.AD_dim in original)
        activation: Final activation (self.conf.G_activation in original)

    Returns:
        GUNet16_2D_BN_with_AD model
    """
    model = GUNet16_2D_BN_with_AD(
        input_shape=input_shape,
        filters=filters,
        latent_space=latent_space,
        age_dim=age_dim,
        AD_dim=AD_dim,
        activation=activation,
    )
    return model


if __name__ == "__main__":
    # Test the exact conversion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create generator with same parameters as original
    generator = create_gan_generator()
    generator = generator.to(device)

    # Test forward pass
    batch_size = 2
    g_input = torch.randn(batch_size, 1, 196, 196).to(device)  # g_input from original
    age_vector = torch.randn(batch_size, 20).to(device)  # age_vector from original
    AD_vector = torch.randn(batch_size, 1).to(device)  # AD_vector from original

    with torch.no_grad():
        output = generator(g_input, age_vector, AD_vector)
        print("\nTest Results:")
        print(f"  Input shape: {g_input.shape}")
        print(f"  Age vector shape: {age_vector.shape}")
        print(f"  AD vector shape: {AD_vector.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
