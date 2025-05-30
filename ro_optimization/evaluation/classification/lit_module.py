# lit_module.py (updated to use config)
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torchmetrics
from .model import ResNetAttrClassifier
from dataset import CelebALMDBDataset
from .data_utils import load_celeba_splits, compute_pos_weight


class AttrClassifier(pl.LightningModule):
    def __init__(self,
                 # Data paths
                 celeba_lmdb: str = None,
                 celeba_attr: str = None,
                 celeba_partition: str = None,
                 celebahq_lmdb: str = None,
                 celebahq_attr: str = None,
                 # Model params
                 img_size: int = 128,
                 backbone: str = 'resnet50',
                 dropout: float = 0.5,
                 # Training params
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 lr_scheduler: str = 'cosine',
                 # Stage selection
                 stage: str = 'celeba',  # 'celeba' or 'celebahq'
                 # Data augmentation
                 use_augmentation: bool = True,
                 # Full config
                 cfg = None):
        
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg or self.hparams.cfg
        
        # Model
        self.model = ResNetAttrClassifier(
            backbone=backbone,
            num_classes=self.cfg.model.num_classes if self.cfg else 40,
            pretrained=self.cfg.model.pretrained if self.cfg else True,
            dropout=dropout
        )
        
        # Metrics
        num_labels = self.cfg.model.num_classes if self.cfg else 40
        self.train_auroc = torchmetrics.AUROC(task='multilabel', num_labels=num_labels)
        self.val_auroc = torchmetrics.AUROC(task='multilabel', num_labels=num_labels)
        
        # Loss weight placeholder
        self.register_buffer('pos_weight', torch.ones(num_labels))
        
    def setup(self, stage=None):
        """Setup datasets based on the training stage."""
        
        if self.hparams.stage == 'celeba':
            # Load CelebA dataset from LMDB
            print("Setting up CelebA dataset from LMDB...")
            
            # Load official splits
            splits = load_celeba_splits(self.hparams.celeba_partition)
            
            # Create train dataset with augmentation
            self.train_dataset = CelebALMDBDataset(
                lmdb_path=self.hparams.celeba_lmdb,
                attr_path=self.hparams.celeba_attr,
                image_size=self.hparams.img_size,
                do_augment=self.hparams.use_augmentation,
                do_normalize=True,
                is_celebahq=False,
                split_files=splits['train']
            )
            
            # Create validation dataset without augmentation
            self.val_dataset = CelebALMDBDataset(
                lmdb_path=self.hparams.celeba_lmdb,
                attr_path=self.hparams.celeba_attr,
                image_size=self.hparams.img_size,
                do_augment=False,
                do_normalize=True,
                is_celebahq=False,
                split_files=splits['val']
            )
            
            print(f"CelebA - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
            
        else:  # celebahq
            print("Setting up CelebA-HQ dataset from LMDB...")
            
            # Create train dataset with augmentation
            self.train_dataset = CelebALMDBDataset(
                lmdb_path=self.hparams.celebahq_lmdb,
                attr_path=self.hparams.celebahq_attr,
                image_size=self.hparams.img_size,
                do_augment=self.hparams.use_augmentation,
                do_normalize=True,
                is_celebahq=True
            )
            
            # Split indices for train/val
            total_size = len(self.train_dataset)
            train_ratio = self.cfg.celebahq.train_split_ratio if self.cfg else 0.9
            train_size = int(train_ratio * total_size)
            
            # Modify valid_indices for train/val split
            self.train_dataset.valid_indices = list(range(train_size))
            
            # Create validation dataset
            self.val_dataset = CelebALMDBDataset(
                lmdb_path=self.hparams.celebahq_lmdb,
                attr_path=self.hparams.celebahq_attr,
                image_size=self.hparams.img_size,
                do_augment=False,
                do_normalize=True,
                is_celebahq=True
            )
            self.val_dataset.valid_indices = list(range(train_size, total_size))
            
            print(f"CelebA-HQ - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        
        # Compute positive weights
        print("Computing class weights...")
        max_weight = self.cfg.loss.pos_weight_cap if self.cfg else 10.0
        self.pos_weight = compute_pos_weight(self.train_dataset, max_weight=max_weight)
    
    def forward(self, x):
        return self.model(x)
    
    def compute_loss(self, logits, labels):
        """Compute weighted BCE loss."""
        return F.binary_cross_entropy_with_logits(
            logits, labels,
            pos_weight=self.pos_weight
        )
    
    def training_step(self, batch, batch_idx):
        imgs = batch['img']
        labels = batch['labels']
        
        logits = self(imgs)
        loss = self.compute_loss(logits, labels)
        
        # Update metrics
        preds = torch.sigmoid(logits)
        self.train_auroc.update(preds, labels.int())
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        # Log AUROC
        auroc = self.train_auroc.compute()
        self.log('train_auroc', auroc, prog_bar=True)
        self.train_auroc.reset()
    
    def validation_step(self, batch, batch_idx):
        imgs = batch['img']
        labels = batch['labels']
        
        logits = self(imgs)
        loss = self.compute_loss(logits, labels)
        
        # Update metrics
        preds = torch.sigmoid(logits)
        self.val_auroc.update(preds, labels.int())
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        # Log AUROC
        auroc = self.val_auroc.compute()
        self.log('val_auroc', auroc, prog_bar=True)
        self.val_auroc.reset()
    
    def configure_optimizers(self):
        # Get optimizer settings from config
        if self.cfg and self.cfg.optim.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay
            )
        else:  # default to adamw
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay
            )
        
        # Configure scheduler
        if self.hparams.lr_scheduler == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.cfg.optim.scheduler.cosine_eta_min if self.cfg else 1e-6
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        elif self.hparams.lr_scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.cfg.optim.scheduler.plateau_factor if self.cfg else 0.5,
                patience=self.cfg.optim.scheduler.plateau_patience if self.cfg else 5,
                min_lr=1e-6
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch'
                }
            }
        else:
            return optimizer
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.cfg.dataloader.pin_memory if self.cfg else True,
            persistent_workers=self.cfg.dataloader.persistent_workers if self.cfg else True,
            drop_last=self.cfg.dataloader.drop_last if self.cfg else True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.cfg.dataloader.pin_memory if self.cfg else True,
            persistent_workers=self.cfg.dataloader.persistent_workers if self.cfg else True
        )