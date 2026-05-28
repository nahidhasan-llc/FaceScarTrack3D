import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# 1. DEFINE A SIMPLIFIED U-NET ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    """(Convolution -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class ToyUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Left Side (Encoder / Down-sampling)
        self.inc = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.down1 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottom Bottleneck
        self.bottleneck = DoubleConv(32, 64)
        
        # Right Side (Decoder / Up-sampling)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(64, 32) # 64 channels because of Skip Connection concatenation
        
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(32, 16) # 32 channels because of Skip Connection concatenation
        
        # Final Output Layer (1x1 Convolution to map to a single binary mask)
        self.outc = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder Path
        x1 = self.inc(x)        # High-res feature map (Saved for Skip Connection 1)
        x2 = self.pool1(x1)
        x3 = self.down1(x2)     # Mid-res feature map (Saved for Skip Connection 2)
        x4 = self.pool2(x3)
        
        # Bottleneck
        b = self.bottleneck(x4)
        
        # Decoder Path with Skip Connections
        temp = self.up1(b)
        merge1 = torch.cat([temp, x3], dim=1) # Paste x3 directly across the U-Net
        x5 = self.up_conv1(merge1)
        
        temp2 = self.up2(x5)
        merge2 = torch.cat([temp2, x1], dim=1) # Paste x1 directly across the U-Net
        x6 = self.up_conv2(merge2)
        
        logits = self.outc(x6)
        return torch.sigmoid(logits) # Squashes output between 0.0 and 1.0 (Probability Map)

# ──────────────────────────────────────────────────────────────────────────────
# 2. GENERATE SYNTHETIC TRAINING DATA (Simulated Burn Scars)
# ──────────────────────────────────────────────────────────────────────────────
def generate_synthetic_data(num_samples=20, size=64):
    """Generates synthetic face images with random red blob 'burn scars'."""
    images = []
    masks = []
    
    for _ in range(num_samples):
        # Create a generic skin tone background image (RGB)
        img = np.zeros((size, size, 3), dtype=np.float32)
        img[:, :, 0] = 0.85  # High Red
        img[:, :, 1] = 0.65  # Mid Green
        img[:, :, 2] = 0.55  # Mid Blue (Creates a flesh/peach tone)
        
        mask = np.zeros((size, size, 1), dtype=np.float32)
        
        # Inject a random circular 'burn scar' region (Intense red/purple inflammation)
        cx, cy = np.random.randint(15, size-15, size=2)
        radius = np.random.randint(6, 12)
        
        for y in range(size):
            for x in range(size):
                if (x - cx)**2 + (y - cy)**2 < radius**2:
                    img[y, x, 0] = 0.95  # Inflamed Deep Red
                    img[y, x, 1] = 0.20  # Drop Green
                    img[y, x, 2] = 0.30  # Drop Blue
                    mask[y, x, 0] = 1.0  # Ground Truth Target Mark
                    
        # Add random background image pixel noise to simulate sensor variations
        img += np.random.normal(0, 0.03, img.shape)
        img = np.clip(img, 0, 1)
        
        # PyTorch expects dimensions: [Channels, Height, Width]
        images.append(np.transpose(img, (2, 0, 1)))
        masks.append(np.transpose(mask, (2, 0, 1)))
        
    return torch.tensor(np.array(images)), torch.tensor(np.array(masks))

# ──────────────────────────────────────────────────────────────────────────────
# 3. TRAINING LOOP PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("Generating synthetic skin datasets...")
    X_train, Y_train = generate_synthetic_data(num_samples=40, size=64)
    X_test, Y_test = generate_synthetic_data(num_samples=1, size=64)
    
    model = ToyUNet()
    criterion = nn.BCELoss() # Binary Cross Entropy Loss (Perfect for 0 or 1 pixel masks)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print("\nTraining Toy U-Net model...")
    model.train()
    for epoch in range(1, 101): # Train for 100 epochs
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/100] -> Pixel Segmentation Loss: {loss.item():.4f}")
            
    # ──────────────────────────────────────────────────────────────────────────
    # 4. EVALUATE AND VISUALIZE THE OUTPUT
    # ──────────────────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        prediction = model(X_test)
        
    # Convert tensors back to standard 2D image matrices for plotting
    raw_img = np.transpose(X_test[0].numpy(), (1, 2, 0))
    true_mask = Y_test[0][0].numpy()
    pred_mask = prediction[0][0].numpy()
    
    print("\nTraining complete! Generating evaluation plot window...")
    
    # Setup visual subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(raw_img)
    axes[0].set_title("Input Patient Image\n(With Simulated Burn)")
    axes[0].axis('off')
    
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title("Ground Truth Mask\n(What it should find)")
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask, cmap='magma')
    axes[2].set_title("U-Net Predicted Output\n(Probability Map)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()