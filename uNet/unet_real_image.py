import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# 1. U-NET ARCHITECTURE DEFINITION
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
    def forward(self, x): 
        return self.conv(x)

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
# 2. DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def load_real_patient_image(image_filename, size=128):
    # Load image and fix color channel ordering
    bgr_img = cv2.imread(image_filename)
    if bgr_img is None:
        raise FileNotFoundError(f"Could not load image at path: {image_filename}")
        
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, (size, size), interpolation=cv2.INTER_AREA)
    normalized_img = resized_img.astype(np.float32) / 255.0
    
    # Isolate red inflammatory pixels to create a basic target mask
    mask = np.zeros((size, size, 1), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            r, g, b = normalized_img[y, x, 0], normalized_img[y, x, 1], normalized_img[y, x, 2]
            if r > 0.55 and g < 0.45:
                mask[y, x, 0] = 1.0  
                
    # Restructure to PyTorch tensor format: [Batch, Channels, Height, Width]
    tensor_img = torch.tensor(np.transpose(normalized_img, (2, 0, 1))).unsqueeze(0)
    tensor_mask = torch.tensor(np.transpose(mask, (2, 0, 1))).unsqueeze(0)
    
    return tensor_img, tensor_mask, normalized_img, mask.squeeze()

# ──────────────────────────────────────────────────────────────────────────────
# 3. PIPELINE RUNTIME EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Set up clean local pathing directly to the asset folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) 
    image_file = os.path.join(project_root, "sample_images/burn_faces", "burn_face_tiff.tif")
    
    X_data, Y_data, original_display, ground_truth_display = load_real_patient_image(image_file, size=128)

    model = ToyUNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    # Simple training loop
    model.train()
    print("Training U-Net model...")
    for epoch in range(1, 151):
        optimizer.zero_grad()
        loss = criterion(model(X_data), Y_data)
        loss.backward()
        optimizer.step()
        
        if epoch % 25 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/150] -> Loss: {loss.item():.4f}")
            
    # Inference execution pass
    model.eval()
    with torch.no_grad():
        prediction = model(X_data)
    predicted_display = prediction[0][0].numpy()
    
    # Render final comparison metrics visual layout
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(original_display)
    axes[0].set_title("Input Patient Image")
    axes[0].axis('off')
    
    axes[1].imshow(ground_truth_display, cmap='gray')
    axes[1].set_title("Target Burn Mask")
    axes[1].axis('off')
    
    axes[2].imshow(predicted_display, cmap='magma')
    axes[2].set_title("U-Net Predicted Output")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()