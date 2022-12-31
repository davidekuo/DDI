import argparse
from torchvision import transforms as T
import wandb

# pytorch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split, DataLoader
from datasets import DDI_Dataset, params_to_list
from models import load_model

# pytorchlightning
import pytorch_lightning as pl
from datasets import DDI_DataModule
from models import DDI_DeepDerm
from pytorch_lightning.loggers import WandbLogger

def pytorch(args, config):
    """DATA PREPARATION"""
    # Read in metadata to DataFrame
    annotations = pd.read_csv(args.annotation_file, index_col=0)

    # Map disease (string) to label (int)
    disease2label = {disease: label for label, disease in enumerate(sorted(annotations['disease'].unique()))}

    # Add 'disease_label' col to annotations with int label for each diagnosis
    annotations['disease_label'] = annotations['disease'].map(disease2label)

    # Parse inputs into lists for downstream processing
    skin_tone = params_to_list(config['skin_tone'], [12, 34, 56])
    malignant = params_to_list(config['malignant'], [True, False])
    diseases = params_to_list(config['diseases'], list(disease2label))

    # Make sure specified inputs are valid
    for s in skin_tone:
        assert s in [12, 34, 56], f"{s} is not a valid skin tone"
    for d in diseases:
        assert d in disease2label, f"{d} is not a valid diagnosis"

    # Select rows of annotations DataFrame that meet specifications
    indices = np.where(annotations['skin_tone'].isin(skin_tone) &
                       annotations['malignant'].isin(malignant) &
                       annotations['disease'].isin(diseases))
    selected_data = annotations.loc[indices]

    # Create DDI_Dataset from selected data
    dataset = DDI_Dataset(selected_data, args.img_dir, transform)

    # Split 60-20-20 (to match DDI) into train/val/test datasets based on random seed
    train_data, val_data, test_data = random_split(dataset,
                                                   [0.6, 0.2, 0.2],
                                                   generator=torch.Generator().manual_seed(args.random_seed))

    # Initialize train/val/test dataloaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)

    """TRAINING SETUP"""
    # Load model
    num_classes = 2 if args.classify_malignant else 78  # 78 diagnoses in DDI
    model = load_model(model_name='DeepDerm', num_classes=num_classes)
    # Set up optimizer
    children_to_fine_tune = {'first_conv': list(model.children())[0],  # Conv2d_1a_3x3
                             'first_block': list(model.children())[0:4],  # include up to maxpool1
                             'before_inception_modules': list(model.children())[0:7],  # ... maxpool2
                             'first_inception_module': list(model.children())[0:8],  # ... 1st inception module
                             }
    params_to_fine_tune = []
    for child in children_to_fine_tune[args.finetune_mode]:
        params_to_fine_tune += [param for param in child.parameters()]
    optimizer = torch.optim.Adam(params_to_fine_tune, lr=1e-4)  # match DDI experiments learning rate

    """TRAINING LOOP"""
    # TODO!!!

def pytorchlightning(args, config):
    # Set random seed
    pl.seed_everything(args.random_seed)

    # Initialize LightningDataModule, LightningModule, Logger, Trainer
    data_module = DDI_DataModule(args.random_seed,
                                 args.batch_size,
                                 args.num_workers,
                                 args.annotation_file,
                                 args.img_dir,
                                 config['transform'],
                                 config['skin_tone'],
                                 config['malignant'],
                                 config['diseases'])
    model = DDI_DeepDerm(classify_malignant=args.classify_malignant,
                         mode=args.finetune_mode)
    wandb_logger = WandbLogger(project='DDI')
    trainer = pl.Trainer(deterministic=True,  # for reproducibility
                         max_epochs=500,  # to match DDI experiments
                         logger=wandb_logger,
                         log_every_n_steps=2)  # 2 batches per epoch

    # Train model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_file', type=str, default='./ddi_data/ddi_metadata.csv')
    parser.add_argument('--img_dir', type=str, default='./ddi_data/')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--classify_malignant', type=bool, default=True)
    parser.add_argument('--finetune_mode', type=str, default='first_conv')
    args = parser.parse_args()

    # Constants and transforms copied from DDI-Code
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    test_transform = T.Compose([
        lambda x: x.convert('RGB'),
        T.Resize(299),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=means, std=stds)
    ])

    # Config
    transform = test_transform  # TODO: match DDI experiments
    skin_tone = None  # [12, 34, 56]
    malignant = None  # [True, False]
    diseases = None
    config = {'transform': transform,
              'skin_tone': skin_tone,
              'malignant': malignant,
              'diseases': diseases,
              }

    # Log into W&B for experiment tracking
    wandb.login()

    # Launch experiment
    pytorchlightning(args, config)
    # pytorch(args, config)
