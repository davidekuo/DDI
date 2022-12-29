import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from ddi_dataset import DDI_DataModule, DDI_Dataset, params_to_list
from ddi_models import DDI_DeepDerm, load_model


def main(args):
	# Parse CLI arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--annotation_file', type=str, default='./ddi_data/ddi_metadata.csv')
	parser.add_argument('--img_dir', type=str, default='./ddi_data/')
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--random_seed', type=int, default=0)
	parser.add_argument('--classify_malignant', type=bool, default=True)
	args = parser.parse_args()

	# Vanilla PyTorch
	### DATA PREPARATION ###

	# Read in metadata to DataFrame
	annotations = pd.read_csv(args.annotation_file, index_col=0)

	# Map disease (string) to label (int)
	disease2label = {disease:label for label, disease in enumerate(sorted(annotations['disease'].unique()))}
	
	# Add 'disease_label' col to annotations with int label for each diagnosis
	annotations['disease_label'] = annotations['disease'].map(disease2label)
	
	# Parse inputs into lists for downstream processing
	skin_tone = params_to_list(skin_tone, [12, 34, 56])
	malignant = params_to_list(malignant, [True, False])
	diseases = params_to_list(diseases, list(disease2label))
	
	# Make sure specified inputs are valid
	for s in skin_tone:
		assert s in [12, 34, 56], f"{s} is not a valid skin tone"
	for d in diseases:
		assert d in disease2label, f"{d} is not a valid diagnosis"
	
	# Select rows of annotations DataFrame that meet specifications
	indices = np.where(annotations['skin_tone'].isin(skin_tone) & \
											annotations['malignant'].isin(malignant) & \
											annotations['disease'].isin(diseases))
	selected_data = annotations.loc[indices]
	
	# Create DDI_Dataset from selected data
	full_dataset = DDI_Dataset(selected_data, args.img_dir, transform)

	# Split 60-20-20 (to match DDI) into train/val/test datasets based on random seed
	train_data, val_data, test_data = random_split(self.dataset, 
                                                     [0.6, 0.2, 0.2], 
                                                     generator=torch.Generator().manual_seed(random_seed))

	# Initialize train/val/test dataloaders
	train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
	val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
	test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)

	### MODEL SETUP ###
	num_classes = 2 if classify_malignant else 78  # 78 diagnoses in DDI
	model = load_model(model_name='DeepDerm', num_classes=num_classes)

	# !!! TO DO !!!
	### TRAINING LOOP ###


if __name__ == "__main__":
	main()
