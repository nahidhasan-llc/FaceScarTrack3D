import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# 1. THE U-NET ENGINE
# ──────────────────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
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
    def forward(self, x): return self.conv(x)

class ToyUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.down1 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(32, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(32, 16)
        self.outc = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.pool1(x1)
        x3 = self.down1(x2)
        x4 = self.pool2(x3)
        b = self.bottleneck(x4)
        
        temp = self.up1(b)
        x5 = self.up_conv1(torch.cat([temp, x3], dim=1))
        temp2 = self.up2(x5)
        x6 = self.up_conv2(torch.cat([temp2, x1], dim=1))
        return torch.sigmoid(self.outc(x6))

# ──────────────────────────────────────────────────────────────────────────────
# 2. FOLDER PREPROCESSING ENGINE
# ──────────────────────────────────────────────────────────────────────────────
def load_entire_folder(folder_path, size=128):
    """Loops through an entire directory, processing all images into batch arrays."""
    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    images_list = []
    masks_list = []
    display_images = []
    display_masks = []
    
    # Loop over every file name in the target folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_extensions):
            file_path = os.path.join(folder_path, filename)
            
            # Load and normalize raw image data
            bgr_img = cv2.imread(file_path)
            if bgr_img is None: continue
            
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(rgb_img, (size, size), interpolation=cv2.INTER_AREA)
            normalized_img = resized_img.astype(np.float32) / 255.0
            
            # Formulate the programmatic target mask layout
            mask = np.zeros((size, size, 1), dtype=np.float32)
            for y in range(size):
                for x in range(size):
                    r, g, b = normalized_img[y, x, 0], normalized_img[y, x, 1], normalized_img[y, x, 2]
                    if r > 0.55 and g < 0.45:
                        mask[y, x, 0] = 1.0
            
            # Append array representations formatted for PyTorch [C, H, W]
            images_list.append(np.transpose(normalized_img, (2, 0, 1)))
            masks_list.append(np.transpose(mask, (2, 0, 1)))
            
            # Save raw arrays separately purely for final matplotlib plotting functions
            display_images.append(normalized_img)
            display_masks.append(mask.squeeze())
            
    if len(images_list) == 0:
        raise FileNotFoundError(f"No valid image files found inside: {folder_path}")
        
    # Stack individual lists into unified multi-image batches [Batch_Size, C, H, W]
    tensor_images = torch.tensor(np.array(images_list))
    tensor_masks = torch.tensor(np.array(masks_list))
    
    return tensor_images, tensor_masks, display_images, display_masks

# ──────────────────────────────────────────────────────────────────────────────
# 3. PIPELINE MATRIX EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    target_folder = os.path.join(project_root, "sample_images/burn_faces")
    
    print(f"Scanning target folder pathway:\n👉 {target_folder}\n")
    X_batch, Y_batch, raw_imgs, raw_masks = load_entire_folder(target_folder, size=128)
    
    num_images = X_batch.shape[0]
    print(f"Successfully compiled batch data matrix totaling: {num_images} images.")

    model = ToyUNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    # Train across the entire combined batch array simultaneously
    model.train()
    print("\nTraining U-Net across entire image folder array...")
    for epoch in range(1, 151):
        optimizer.zero_grad()
        loss = criterion(model(X_batch), Y_batch)
        loss.backward()
        optimizer.step()
        
        if epoch % 25 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/150] -> Batch Loss: {loss.item():.4f}")
            
    # Run evaluation pass across everything
    model.eval()
    with torch.no_grad():
        predictions = model(X_batch).numpy()
        
    print("\nProcessing complete! Constructing multi-image dashboard layout...")
    
    # Dynamic Plot Grid: 1 row per image, 3 columns per row (Input, Target, Prediction)
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 3 * num_images))
    
    # If there is only 1 image in the folder, wrap axes array so indexing loops work cleanly
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for idx in range(num_images):
        # Column 1: Input Photograph
        axes[idx, 0].imshow(raw_imgs[idx])
        axes[idx, 0].set_title(f"Img {idx+1}: Original") if idx == 0 else axes[idx, 0].set_title(f"Img {idx+1}")
        axes[idx, 0].axis('off')
        
        # Column 2: Ground Truth Goal Mask
        axes[idx, 1].imshow(raw_masks[idx], cmap='gray')
        axes[idx, 1].set_title("Target Mask") if idx == 0 else None
        axes[idx, 1].axis('off')
        
        # Column 3: Network Inference
        axes[idx, 2].imshow(predictions[idx][0], cmap='magma')
        axes[idx, 2].set_title("U-Net Prediction") if idx == 0 else None
        axes[idx, 2].axis('off')
        
    plt.tight_layout()
    print("Launching comparison grid window...")
    plt.show()

if __name__ == "__main__":
    main()