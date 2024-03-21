# This is unet_config dictionary contains the following parameters:

# sample_size: The spatial size of the input and output tensors. In this example, it is set to 32, meaning the input and output tensors will have a spatial size of 32x32.
# in_channels: The number of input channels. It is set to 1, assuming the input is a single-channel spectrogram.
# out_channels: The number of output channels. It is set to 1, assuming the output is a single-channel spectrogram.
# layers_per_block: The number of layers per block in the UNet architecture. It is set to 2, meaning each block will have 2 layers.
# block_out_channels: A tuple specifying the number of output channels for each block in the UNet. In this example, it is set to (32, 64, 128, 256), meaning the blocks will have 32, 64, 128, and 256 output channels, respectively.
# down_block_types: A tuple specifying the type of down blocks to use in the UNet. In this example, it is set to ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"), meaning all down blocks will be of type "DownBlock2D".
# up_block_types: A tuple specifying the type of up blocks to use in the UNet. In this example, it is set to ("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"), meaning all up blocks will be of type "UpBlock2D".
# latent_channels: The number of channels in the latent space of the UNet. It is set to 256, meaning the latent space will have 256 channels.
# cross_attention_dim: The dimensionality of the cross-attention layers used in the UNet. It is set to 256, meaning the cross-attention layers will have a dimensionality of 256.
# attention_head_dim: A tuple specifying the dimensionality of the attention heads in each block of the UNet. In this example, it is set to (8, 8, 8, 8), meaning each block will have attention heads with a dimensionality of 8.

unet_config = {
    "sample_size": 32,  # The spatial size of the input/output tensors
    "in_channels": 1,  # The number of input channels
    "out_channels": 1,  # The number of output channels
    "layers_per_block": 2,  # The number of layers per block in the UNet
    "block_out_channels": (32, 64, 128, 256),  # The number of output channels for each block
    "down_block_types": (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
    ),  # The type of down blocks to use
    "up_block_types": (
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),  # The type of up blocks to use
    "latent_channels": 256,  # The number of channels in the latent space
    "cross_attention_dim": 256,  # The dimensionality of the cross-attention layers
    "attention_head_dim": (8, 8, 8, 8),  # The dimensionality of the attention heads in each block
}