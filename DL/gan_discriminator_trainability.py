"""
GAN Discriminator Trainability Exploration
Compares: Discriminator always trainable vs always frozen
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== ARCHITECTURE ====================

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x.view(-1, 28 * 28))

# ==================== TRAINING FUNCTION ====================

def train_gan(discriminator_trainable=True, epochs=50, batch_size=64, latent_dim=100):
    """
    Train GAN with specified discriminator trainability setting
    
    Args:
        discriminator_trainable: If True, discriminator trains every step
                                If False, discriminator is frozen
        epochs: Number of epochs
        batch_size: Batch size
        latent_dim: Latent dimension for generator
    
    Returns:
        dict with training history
    """
    
    print(f"\n{'='*60}")
    print(f"Training with Discriminator Trainable = {discriminator_trainable}")
    print(f"{'='*60}\n")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize models
    G = Generator(latent_dim).to(device)
    D = Discriminator().to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Loss
    criterion = nn.BCELoss()
    
    # Set discriminator trainability
    D.requires_grad_(discriminator_trainable)
    
    # History
    history = {
        'g_loss': [],
        'd_loss': [],
        'real_pred': [],
        'fake_pred': []
    }
    
    # Training loop
    for epoch in range(epochs):
        g_loss_epoch = 0
        d_loss_epoch = 0
        real_pred_epoch = []
        fake_pred_epoch = []
        
        for batch_idx, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(device)
            batch_size_actual = real_images.size(0)
            
            # Labels
            real_labels = torch.ones(batch_size_actual, 1).to(device)
            fake_labels = torch.zeros(batch_size_actual, 1).to(device)
            
            # ===== Train Discriminator =====
            if discriminator_trainable:
                optimizer_D.zero_grad()
                
                # Real images
                real_output = D(real_images)
                d_loss_real = criterion(real_output, real_labels)
                
                # Fake images
                z = torch.randn(batch_size_actual, latent_dim).to(device)
                fake_images = G(z).detach()
                fake_output = D(fake_images)
                d_loss_fake = criterion(fake_output, fake_labels)
                
                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_D.step()
                
                d_loss_epoch += d_loss.item()
                real_pred_epoch.append(real_output.mean().item())
                fake_pred_epoch.append(fake_output.mean().item())
            
            # ===== Train Generator =====
            optimizer_G.zero_grad()
            
            z = torch.randn(batch_size_actual, latent_dim).to(device)
            fake_images = G(z)
            fake_output = D(fake_images)
            
            # Generator tries to fool discriminator
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizer_G.step()
            
            g_loss_epoch += g_loss.item()
        
        # Average losses
        g_loss_avg = g_loss_epoch / len(train_loader)
        d_loss_avg = d_loss_epoch / len(train_loader) if discriminator_trainable else 0
        real_pred_avg = np.mean(real_pred_epoch) if real_pred_epoch else 0
        fake_pred_avg = np.mean(fake_pred_epoch) if fake_pred_epoch else 0
        
        history['g_loss'].append(g_loss_avg)
        history['d_loss'].append(d_loss_avg)
        history['real_pred'].append(real_pred_avg)
        history['fake_pred'].append(fake_pred_avg)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  G Loss: {g_loss_avg:.4f}")
            if discriminator_trainable:
                print(f"  D Loss: {d_loss_avg:.4f}")
                print(f"  D(Real): {real_pred_avg:.4f}, D(Fake): {fake_pred_avg:.4f}")
            else:
                print(f"  D is FROZEN (not training)")
    
    return history, G, D

# ==================== COMPARISON ====================

if __name__ == "__main__":
    # Train both scenarios
    history_trainable, G_trainable, D_trainable = train_gan(
        discriminator_trainable=True, 
        epochs=50
    )
    
    history_frozen, G_frozen, D_frozen = train_gan(
        discriminator_trainable=False, 
        epochs=50
    )
    
    # ===== PLOTTING =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Generator Loss Comparison
    axes[0, 0].plot(history_trainable['g_loss'], label='D Trainable', linewidth=2)
    axes[0, 0].plot(history_frozen['g_loss'], label='D Frozen', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Generator Loss')
    axes[0, 0].set_title('Generator Loss: Trainable vs Frozen Discriminator')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Discriminator Loss (only trainable scenario)
    axes[0, 1].plot(history_trainable['d_loss'], label='D Loss', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Discriminator Loss')
    axes[0, 1].set_title('Discriminator Loss (Trainable Scenario Only)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Discriminator Predictions (trainable)
    axes[1, 0].plot(history_trainable['real_pred'], label='D(Real)', linewidth=2)
    axes[1, 0].plot(history_trainable['fake_pred'], label='D(Fake)', linewidth=2)
    axes[1, 0].axhline(0.5, color='red', linestyle='--', label='Baseline (0.5)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Discriminator Output')
    axes[1, 0].set_title('D Predictions: Trainable Scenario')
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Discriminator Predictions (frozen)
    axes[1, 1].plot(history_frozen['real_pred'], label='D(Real)', linewidth=2)
    axes[1, 1].plot(history_frozen['fake_pred'], label='D(Fake)', linewidth=2)
    axes[1, 1].axhline(0.5, color='red', linestyle='--', label='Baseline (0.5)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Discriminator Output')
    axes[1, 1].set_title('D Predictions: Frozen Scenario')
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/shk/.openclaw/workspace/gan_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved to: gan_comparison.png")
    plt.show()
    
    # ===== ANALYSIS =====
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    print("\n📊 TRAINABLE DISCRIMINATOR:")
    print(f"  Final G Loss: {history_trainable['g_loss'][-1]:.4f}")
    print(f"  Final D Loss: {history_trainable['d_loss'][-1]:.4f}")
    print(f"  D(Real) → D(Fake) gap: {abs(history_trainable['real_pred'][-1] - history_trainable['fake_pred'][-1]):.4f}")
    print(f"  ➜ Generator struggles because D keeps improving")
    
    print("\n🧊 FROZEN DISCRIMINATOR:")
    print(f"  Final G Loss: {history_frozen['g_loss'][-1]:.4f}")
    print(f"  D never trains (frozen weights)")
    print(f"  ➜ Generator's feedback signal is meaningless")
    
    print("\n💡 KEY OBSERVATION:")
    if history_trainable['g_loss'][-1] > history_frozen['g_loss'][-1]:
        print("  G loss is HIGHER with trainable D: training is harder/unstable")
    else:
        print("  G loss is LOWER with trainable D: but likely poor sample quality")
    
    print("\n✅ Script complete! Check the plots for visual comparison.")
