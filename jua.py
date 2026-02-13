import os
import random
import argparse
import numpy as np
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

# k-NN evaluation
from sklearn.neighbors import KNeighborsClassifier
import wandb

# =============================================================================
# 1. Configuration & Parsing
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Generic SSL Training Script")
    
    # Data & Paths
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training dataset (ImageFolder structure)")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to validation dataset. If None, uses train set for eval.")
    parser.add_argument("--wandb_key", type=str, default=None, help="WandB API Key")
    parser.add_argument("--project_name", type=str, default="ssl-generic", help="WandB project name")
    
    # Training Hyperparams
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=32, help="Input image size")
    
    # Model & Loss
    parser.add_argument("--loss", type=str, choices=["vae", "flow"], default="vae", help="Choose loss function")
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--small_conv", action="store_true", help="Use 3x3 conv1 (optimized for small images like CIFAR)")
    
    # Evaluation
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--linear_probe_epochs", type=int, default=100)
    parser.add_argument("--knn_k", type=int, default=20)

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# 2. Data Augmentation & Dataset
# =============================================================================
class MultiViewTransform:
    def __init__(self, transform, num_views=2):
        self.transform = transform
        self.num_views = num_views

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_views)]

def get_transforms(img_size):
    # Calculate odd kernel size
    kernel_size = max(3, img_size // 20)
    if kernel_size % 2 == 0:  # Make it odd
        kernel_size += 1
    
    ssl_transform = T.Compose([
        T.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(0.5),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=kernel_size)], p=0.5),
        T.RandomSolarize(threshold=128, p=0.2),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return ssl_transform

    test_transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return ssl_transform, test_transform

# =============================================================================
# 3. Model Architecture
# =============================================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        freq_scale = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -freq_scale)
        t_proj = t * embeddings
        return torch.cat((t_proj.sin(), t_proj.cos()), dim=-1)

class ResNetEncoder(nn.Module):
    def __init__(self, small_conv=False):
        super().__init__()
        base_model = resnet18(weights=None)
        
        if small_conv:
            base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base_model.maxpool = nn.Identity()
            
        self.backbone = base_model
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)

class NonLinearPredictionHead1(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2046, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.mean = nn.Linear(output_dim, output_dim)
        self.std = nn.Linear(output_dim, output_dim)
        
    def forward(self, x1, context):
        x = torch.cat([x1, context], dim=1)
        x = self.net(x)
        return self.mean(x), self.std(x)

class NonLinearPredictionHead2(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2046, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x1, context):
        x = torch.cat([x1, context], dim=1)
        return self.net(x)

class VelocityPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, time_emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, y_t, context, t_emb):
        x = torch.cat([y_t, context, t_emb], dim=1)
        return self.net(x)

class GenericSSLModel(nn.Module):
    def __init__(self, embedding_dim=512, loss_type="vae", small_conv=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.loss_type = loss_type
        
        self.backbone = ResNetEncoder(small_conv=small_conv)
        self.projection_head2 = NonLinearPredictionHead2(output_dim=embedding_dim)
        self.projection_head1 = NonLinearPredictionHead1(output_dim=embedding_dim)
        
        time_emb_dim = 128
        self.time_embedding = SinusoidalTimeEmbedding(dim=time_emb_dim)
        self.velocity_predictor = VelocityPredictor(
            input_dim=embedding_dim * 2,
            hidden_dim=1024,
            output_dim=embedding_dim,
            time_emb_dim=time_emb_dim,
        )
        self.flow_projection = nn.Sequential(
             nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Linear(1024, embedding_dim)
        )

    def forward(self, views):
        """
        VAE branch  → returns (recon_loss + kl_loss)  as a scalar (unchanged from original).
        Flow branch → returns (preds, truths) as (2N, D) vectors so the training
                      loop can call F.mse_loss(preds, truths, ...) on gathered tensors.

        The training loop checks self.loss_type (or isinstance) to decide how to
        reduce the output into a final scalar for .backward().
        """
        view1, view2 = views[0], views[1]

        online_feat1 = self.backbone(view1)
        online_feat2 = self.backbone(view2)

        if self.loss_type == "vae":
            # ---- original VAE logic, untouched ----
            mean, logvar = self.projection_head1(online_feat1.detach(), online_feat2.detach())
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            sample = mean + eps * std

            pred = self.projection_head2(online_feat1, sample)

            kl  = 1 + logvar - mean.pow(2) - logvar.exp()

            return  pred, online_feat2, kl

        elif self.loss_type == "flow":
            context1 = self.flow_projection(online_feat1)
            context2 = self.flow_projection(online_feat2)

            target1 = online_feat1.detach()
            target2 = online_feat2.detach()

            # View 1 -> View 2
            y_0 = torch.randn_like(target2)
            t = torch.rand(target2.shape[0], 1, device=target2.device)
            y_t = t * target2 + (1 - t) * y_0
            true_velocity_12 = target2 - y_0
            t_emb = self.time_embedding(t)
            pred_velocity_12 = self.velocity_predictor(y_t, context1, t_emb)

            # View 2 -> View 1
            y_0_b = torch.randn_like(target1)
            t_b = torch.rand(target1.shape[0], 1, device=target1.device)
            y_t_b = t_b * target1 + (1 - t_b) * y_0_b
            true_velocity_21 = target1 - y_0_b
            t_emb_b = self.time_embedding(t_b)
            pred_velocity_21 = self.velocity_predictor(y_t_b, context2, t_emb_b)

            preds  = torch.cat([pred_velocity_21, pred_velocity_12], dim=0)  # (2N, D)
            truths = torch.cat([true_velocity_21, true_velocity_12], dim=0)  # (2N, D)

            return preds, truths

# =============================================================================
# 4. Evaluation
# =============================================================================
def evaluate_model(model, train_loader, test_loader, device, args, num_classes):
    print("\n--- Starting Evaluation ---")
    was_training = model.training
    model.eval()
    
    backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone

    def extract_features(loader, desc):
        feats, labels = [], []
        with torch.no_grad():
            for x, y in tqdm(loader, desc=desc):
                x = x.to(device)
                f = backbone(x)
                feats.append(f.cpu())
                labels.append(y.cpu())
        return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

    train_feats, train_labels = extract_features(train_loader, "Extract Train Features")
    test_feats, test_labels   = extract_features(test_loader,  "Extract Test Features")

    # k-NN
    print("Running k-NN...")
    knn_train = F.normalize(train_feats, dim=1).numpy()
    knn_test  = F.normalize(test_feats,  dim=1).numpy()
    knn = KNeighborsClassifier(n_neighbors=args.knn_k, n_jobs=-1)
    knn.fit(knn_train, train_labels.numpy())
    knn_acc = knn.score(knn_test, test_labels.numpy()) * 100.0
    print(f"--- k-NN Accuracy: {knn_acc:.2f}% ---")
    if wandb.run: wandb.log({"eval/knn_accuracy": knn_acc})

    # Linear Probe
    print("Training Linear Probe...")
    probe = nn.Linear(train_feats.shape[1], num_classes).to(device)
    opt_probe = optim.AdamW(probe.parameters(), lr=5e-3)
    crit = nn.CrossEntropyLoss()
    
    probe_ds     = TensorDataset(train_feats.to(device), train_labels.to(device))
    probe_loader = DataLoader(probe_ds, batch_size=args.batch_size, shuffle=True)
    
    for _ in range(args.linear_probe_epochs):
        for f_batch, y_batch in probe_loader:
            opt_probe.zero_grad()
            loss = crit(probe(f_batch), y_batch)
            loss.backward()
            opt_probe.step()
            
    with torch.no_grad():
        logits = probe(test_feats.to(device))
        preds  = logits.argmax(dim=1)
        acc    = (preds.cpu() == test_labels).float().mean().item() * 100.0
        
    print(f"--- Linear Probe Accuracy: {acc:.2f}% ---")
    if wandb.run: wandb.log({"eval/linear_probe_accuracy": acc})

    if was_training: model.train()

# =============================================================================
# 5. Main
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Loss: {args.loss}")

    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    wandb.init(project=args.project_name, config=vars(args))

    ssl_aug, test_aug = get_transforms(args.img_size)
    
    print(f"Loading Training Data from: {args.train_dir}")
    train_ds = ImageFolder(args.train_dir, transform=MultiViewTransform(ssl_aug, num_views=2))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    eval_dir = args.val_dir if args.val_dir else args.train_dir
    print(f"Loading Eval Data from: {eval_dir}")
    
    train_eval_ds = ImageFolder(args.train_dir, transform=test_aug)
    test_eval_ds  = ImageFolder(eval_dir,       transform=test_aug)
    
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_eval_loader  = DataLoader(test_eval_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    num_classes = len(train_ds.classes)
    print(f"Detected {num_classes} classes.")

    model = GenericSSLModel(
        embedding_dim=args.embedding_dim,
        loss_type=args.loss,
        small_conv=args.small_conv
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_epochs  = args.epochs
    warmup_epochs = 10

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    print("--- Starting Training ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for views, _ in pbar:
            views = [v.to(device) for v in views]

            optimizer.zero_grad()

            output = model(views)

            if args.loss == "vae":
                # VAE returns a scalar loss directly (original behaviour).
                # DataParallel may gather it into a 1-D vector, so reduce with .mean().
                pred , truth, vvect = output 
                vloss =  -0.5 * torch.mean(vvect)
                loss = F.mse_loss(pred, truth.detach()) + 1.0*vloss
            else:
                # Flow returns (preds, truths) vectors; DataParallel gathers along
                # dim-0, so we still get clean (N, D) tensors here.
                preds, truths = output
                loss = F.mse_loss(preds, truths)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            wandb.log({"train/step_loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")
        wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch, "lr": scheduler.get_last_lr()[0]})

        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            evaluate_model(model, train_eval_loader, test_eval_loader, device, args, num_classes)

    wandb.finish()

if __name__ == "__main__":
    main()
