import torch
import torch.nn as nn
import torch.nn.functional as F

# Model version to ensure we're using the updated architecture
MODEL_VERSION = "v3.0_deeper_discriminator"


class ConvReLU(nn.Module):
    """Conv2D + ReLU (matching conv2D_layer from original)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.relu(self.conv(x))


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


class Critic2D_with_ASD(nn.Module):
    """
    Exact PyTorch conversion of critic_2D_with_AD from discriminator.py
    Line-by-line conversion from Keras to PyTorch
    """

    def __init__(
        self,
        input_shape=(1, 196, 196),
        filters=32,
        age_dim=20,
        ASD_dim=1,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.filters = filters
        self.age_dim = age_dim
        self.ASD_dim = ASD_dim

        f = filters
        # ========== DISCRIMINATOR PATH (exact match to original) ==========
        # Line 22-23: conv1_1
        self.conv1_1 = ConvReLU(input_shape[0], f, kernel_size=3, stride=1, padding=1)

        # Line 24: pool1 = MaxPool2D()(conv1_1)
        self.pool1 = nn.MaxPool2d(2)

        # Line 27-28: conv2_1
        self.conv2_1 = ConvReLU(f, f * 2, kernel_size=3, stride=1, padding=1)

        # Line 29: pool2 = MaxPool2D()(conv2_1)
        self.pool2 = nn.MaxPool2d(2)

        # Line 32-35: conv3_1 and conv3_2
        self.conv3_1 = ConvReLU(f * 2, f * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = ConvReLU(f * 4, f * 4, kernel_size=3, stride=1, padding=1)

        # Line 36: pool3 = MaxPool2D()(conv3_2)
        self.pool3 = nn.MaxPool2d(2)

        # Line 39-42: conv4_1 and conv4_2
        self.conv4_1 = ConvReLU(f * 4, f * 8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = ConvReLU(f * 8, f * 8, kernel_size=3, stride=1, padding=1)

        # Line 43: pool4 = MaxPool2D()(conv4_2)
        self.pool4 = nn.MaxPool2d(2)

        # ========== DEEPER ARCHITECTURE WITH ADDITIONAL LAYERS ==========
        # conv5_1 and conv5_2 - input f*8=256, output f*16=512 (making it deeper)
        self.conv5_1 = ConvReLU(f * 8, f * 16, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = ConvReLU(f * 16, f * 16, kernel_size=3, stride=1, padding=1)
        # pool5 for downsampling
        self.pool5 = nn.MaxPool2d(2)

        # conv6_1 and conv6_2 - input f*16=512, output f*16=512 (more depth)
        self.conv6_1 = ConvReLU(f * 16, f * 16, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = ConvReLU(f * 16, f * 16, kernel_size=3, stride=1, padding=1)
        # pool6 for final downsampling
        self.pool6 = nn.MaxPool2d(2)

        # ========== MIDDLE LAYER WITH CONDITIONING (adjusted for deeper architecture) ==========
        # Line 48-49: mid1_1 - connect to conv6 output (f*16 = 512 channels)
        self.mid1_1 = ConvBNReLU(f * 16, f, kernel_size=3, stride=1, padding=1)

        # Calculate flattened size (matching comment: batch size, 4160)
        self.flatten_size = self._calculate_flatten_size()

        # Line 53-54: dens1_1 = Dense(units=130, activation="sigmoid") + BatchNormalization
        self.dens1_1 = nn.Linear(self.flatten_size, 130)
        self.bn1_1 = nn.BatchNorm1d(130)

        # Line 61-62: dens2_1 = Dense(units=130, activation="relu") + BatchNormalization
        self.dens2_1 = nn.Linear(130 + age_dim, 130)
        self.bn2_1 = nn.BatchNorm1d(130)

        # Line 67: dens3_1 = Dense(units=4160, activation="relu")
        self.dens3_1 = nn.Linear(130 + ASD_dim, self.flatten_size)

        # ========== FINAL LAYERS (updated for deeper architecture) ==========
        # Line 74-77: convF_1 and convF_2 - input is now f*16 + f from concatenation
        self.convF_1 = ConvReLU(f * 16 + f, f * 32, kernel_size=3, stride=1, padding=1)
        self.convF_2 = ConvReLU(f * 32, f * 32, kernel_size=3, stride=1, padding=1)

        # Line 80-83: convD_1 and convD_2 with spectral normalization for stability
        self.convD_1 = ConvReLU(f * 32, f * 32, kernel_size=3, stride=1, padding=1)
        self.convD_2 = nn.utils.spectral_norm(
            nn.Conv2d(f * 32, 1, kernel_size=3, stride=1, padding=1)
        )

        # Line 85: averagePool = GlobalAveragePooling2D()(convD_2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Initialize final layer with smaller weights for stability
        nn.init.normal_(self.convD_2.weight, 0.0, 0.02)
        if self.convD_2.bias is not None:
            nn.init.zeros_(self.convD_2.bias)

    def _calculate_flatten_size(self):
        """Calculate flattened size after 6 pooling operations"""
        with torch.no_grad():
            x = torch.zeros(1, *self.input_shape)
            # Apply 6 pooling operations for deeper architecture
            for _ in range(6):
                x = F.max_pool2d(x, 2)
            # Calculate the spatial size after 6 poolings and apply filters
            spatial_size = x.shape[2] * x.shape[3]
            return self.filters * spatial_size

    def forward(self, d_input, age_vector, ASD_vector):
        """
        Forward pass - exact conversion from original Keras model

        Args:
            d_input: Input image tensor (line 19: d_input = Input(inp_shape))
            age_vector: Age vector (line 55: age_vector = Input(shape=(self.conf.age_dim,)))
            AD_vector: AD vector (line 64: AD_vector = Input(shape=(self.conf.AD_dim,)))
        """

        # ========== DISCRIMINATOR PATH ==========
        # Line 22-24: First conv and pool
        conv1_1 = self.conv1_1(d_input)
        pool1 = self.pool1(conv1_1)  # (batch size, 80, 104, filters)

        # Line 27-29: Second conv and pool
        conv2_1 = self.conv2_1(pool1)
        pool2 = self.pool2(conv2_1)  # (batch size, 40, 52, filters*2)

        # Line 32-36: Third conv layers and pool
        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        pool3 = self.pool3(conv3_2)  # (batch size, 20, 26, filters*4)

        # Line 39-43: Fourth conv layers and pool
        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        pool4 = self.pool4(conv4_2)  # (batch size, 10, 13, filters*8)

        # ========== NEW FIFTH AND SIXTH CONV LAYERS (DEEPER) ==========
        conv5_1 = self.conv5_1(pool4)  # f*8 -> f*16
        conv5_2 = self.conv5_2(conv5_1)  # f*16 -> f*16
        pool5 = self.pool5(conv5_2)  # (batch size, smaller size, filters*16)

        # Sixth layer for even more depth
        conv6_1 = self.conv6_1(pool5)  # f*16 -> f*16
        conv6_2 = self.conv6_2(conv6_1)  # f*16 -> f*16  
        pool6 = self.pool6(conv6_2)  # (batch size, very small size, filters*16)

        # ========== MIDDLE LAYER WITH CONDITIONING ==========
        # Line 48-49: Middle layer - connect to pool6 (deepest layer)
        mid1_1 = self.mid1_1(pool6)  # (batch size, very small size, filter)

        # Line 51: flat1_1 = Flatten()(mid1_1)
        flat1_1 = mid1_1.view(mid1_1.size(0), -1)  # (batch size, 4160)

        # Line 53-54: dens1_1 = Dense(..., activation="sigmoid") + BatchNormalization
        dens1_1 = torch.sigmoid(self.bn1_1(self.dens1_1(flat1_1)))

        # Line 58: mid_concat1_1 = Concatenate()([dens1_1, age_vector])
        # Ensure age_vector is on the same device as dens1_1
        age_vector = age_vector.to(dens1_1.device)
        mid_concat1_1 = torch.cat(
            [dens1_1, age_vector], dim=1
        )  # (batch_size, 130+age_dim)

        # Line 61-62: dens2_1 = Dense(..., activation="relu") + BatchNormalization
        dens2_1 = F.relu(self.bn2_1(self.dens2_1(mid_concat1_1)))

        # Line 65: mid_concat2_1 = Concatenate()([dens2_1, ASD_vector])
        # Ensure ASD_vector is on the same device as dens2_1
        ASD_vector = ASD_vector.to(dens2_1.device)
        mid_concat2_1 = torch.cat(
            [dens2_1, ASD_vector], dim=1
        )  # (batch size, 130+ASD_dim)

        # Line 67: dens3_1 = Dense(units=flatten_size, activation="relu")
        dens3_1 = F.relu(self.dens3_1(mid_concat2_1))  # (batch size, flatten_size)
        # Line 69: rshape1_1 = Reshape - calculate correct spatial dimensions
        batch_size = dens3_1.size(0)
        total_elements = dens3_1.size(1)  # This should match self.flatten_size
        spatial_elements = total_elements // self.filters  # Total spatial pixels

        # The spatial dimensions should match pool6, but let's calculate from actual data
        pool6_h, pool6_w = pool6.shape[2], pool6.shape[3]
        expected_spatial = pool6_h * pool6_w

        if spatial_elements != expected_spatial:
            # Use the actual spatial dimensions from dens3_1 output
            # Try to make it as close to square as possible
            spatial_h = int(spatial_elements**0.5)
            spatial_w = spatial_elements // spatial_h
            # If not perfectly divisible, adjust
            while spatial_h * spatial_w != spatial_elements and spatial_h > 1:
                spatial_h -= 1
                spatial_w = spatial_elements // spatial_h
        else:
            spatial_h, spatial_w = pool6_h, pool6_w

        rshape1_1 = dens3_1.view(
            batch_size, self.filters, spatial_h, spatial_w
        )  # (batch size, filters, spatial_h, spatial_w)

        # If rshape1_1 doesn't match pool6 dimensions, we need to resize it
        if rshape1_1.shape[2:] != pool6.shape[2:]:
            rshape1_1 = F.interpolate(
                rshape1_1, size=pool6.shape[2:], mode="bilinear", align_corners=False
            )

        # Line 71: mid_concat3_1 = Concatenate()([pool6, rshape1_1])
        mid_concat3_1 = torch.cat(
            [pool6, rshape1_1], dim=1
        )  # (batch size, spatial_h, spatial_w, 32+f*16)

        # ========== FINAL DISCRIMINATOR LAYERS ==========
        # Line 74-77: convF_1 and convF_2
        convF_1 = self.convF_1(mid_concat3_1)
        convF_2 = self.convF_2(convF_1)  # (batch size, very small, filters*32)

        # Line 80-83: convD_1 and convD_2
        convD_1 = self.convD_1(convF_2)
        convD_2 = self.convD_2(convD_1)  # (batch size, 10, 13, 1)

        # Line 85: averagePool = GlobalAveragePooling2D()(convD_2)
        averagePool = self.global_avg_pool(convD_2)  # (batch size, 1, 1, 1)
        output = averagePool.view(averagePool.size(0), -1)  # (batch size, 1)

        return output


def create_discriminator(
    input_shape=(1, 196, 196),  # Single channel input (no conditional channels)
    filters=32,
    age_dim=20,
    ASD_dim=1,
):
    """
    Factory function to create discriminator - exact match to original build() method

    Args:
        input_shape: Input image shape (channels, height, width)
        filters: Base number of filters (f in original)
        age_dim: Age vector dimension (self.conf.age_dim in original)
        AD_dim: AD vector dimension (self.conf.AD_dim in original)

    Returns:
        Critic2D_with_ASD model
    """
    model = Critic2D_with_ASD(
        input_shape=input_shape,
        filters=filters,
        age_dim=age_dim,
        ASD_dim=ASD_dim,
    )

    print("Created Deeper Discriminator (6-layer architecture):") 
    print(f"  Input shape: {input_shape[1:]}")
    print(f"  Filters: {filters}")
    print(f"  Age dim: {age_dim}")
    print(f"  AD dim: {ASD_dim}")
    print(f"  Architecture: 6 conv layers + conditional processing")

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator = create_discriminator()
    discriminator = discriminator.to(device)

    batch_size = 2
    d_input = torch.randn(batch_size, 1, 196, 196).to(device)
    age_vector = torch.randn(batch_size, 20).to(device)
    ASD_vector = torch.randn(batch_size, 1).to(device)

    with torch.no_grad():
        output = discriminator(d_input, age_vector, ASD_vector)
        print(f"\nTest Results:")
        print(f"  Input shape: {d_input.shape}")
        print(f"  Age vector shape: {age_vector.shape}")
        print(f"  ASD vector shape: {ASD_vector.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
