import os
import torch
import torchvision
import gdown
import pytorch_lightning as pl

"""Copied from DDI-Code"""
# google drive paths to our models
MODEL_WEB_PATHS = {
    # base form of models trained on skin data
    'HAM10000': 'https://drive.google.com/uc?id=1ToT8ifJ5lcWh8Ix19ifWlMcMz9UZXcmo',
    'DeepDerm': 'https://drive.google.com/uc?id=1OLt11htu9bMPgsE33vZuDiU5Xe4UqKVJ',

    # robust training algorithms
    'GroupDRO': 'https://drive.google.com/uc?id=193ippDUYpMaOaEyLjd1DNsOiW0aRXL75',
    'CORAL': 'https://drive.google.com/uc?id=18rMU0nRd4LiHN9WkXoDROJ2o2sG1_GD8',
    'CDANN': 'https://drive.google.com/uc?id=1PvvgQVqcrth840bFZ3ddLdVSL7NkxiRK',
}

# thresholds determined by maximizing F1-score on the test split of the train 
# dataset for the given algorithm
MODEL_THRESHOLDS = {
    'HAM10000': 0.733,
    'DeepDerm': 0.687,
    # robust training algorithms
    'GroupDRO': 0.980,
    'CORAL': 0.990,
    'CDANN': 0.980,
}


def load_model(model_name, save_dir='./model', num_classes=78):
    # download trained DDI model state_dict to save_dir
    os.makedirs(save_dir, exist_ok=True)
    model_path = save_dir + os.sep + f"{model_name.lower()}.pth"
    gdown.download(MODEL_WEB_PATHS[model_name], model_path)

    # instantiate model with num_outputs
    model = torchvision.models.inception_v3(pretrained=False,
                                            transform_input=True,  # ???
                                            num_classes=num_classes)
    # model.fc = torch.nn.Linear(768, num_outputs) # 78 diagnoses
    # model.AuxLogits.fc = torch.nn.Linear(2048, num_outputs) # 78 diagnoses
    # ^^^ no longer needed w/ num_classes=num_classes ^^^

    # load trained model parameters
    model.load_state_dict(torch.load(model_path))

    # ??? not sure what these are for ???
    model._ddi_name = model_name
    model._ddi_threshold = MODEL_THRESHOLDS[model_name]
    model._ddi_web_path = MODEL_WEB_PATHS[model_name]

    return model


class DDI_DeepDerm(pl.LightningModule):
    """Lightning Module for DDI DeepDerm Model"""

    def __init__(self, classify_malignant: bool = True, mode: str = 'first_conv'):
        super().__init__()
        self.num_classes = 2 if classify_malignant else 78  # 78 diagnoses in DDI
        self.mode = mode
        self.model = load_model(model_name='DeepDerm',
                                num_classes=self.num_classes)

    def training_step(self, batch, batch_idx):
        image, skin_tone, malignant, disease, disease_label = batch
        output = self(image)
        if self.num_classes == 2:  # binary classification: malignant vs. benign
            loss = torch.nn.functional.binary_cross_entropy(output, malignant)
            # ??? https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html
        else:  # multi-class classification (78 diagnoses)
            loss = torch.nn.functional.cross_entropy(output, disease_label)
            # ??? https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
        return loss

    def configure_optimizers(self):
        """
    Inception_v3: https://github.com/pytorch/vision/blob/32d254bbfcf14975f846765775584e61ef25a5bc/torchvision/models/inception.py#L103
    Conv2d_1a_3x3 -> Conv2d_2a_3x3 -> Conv2d_2b_3x3 -> maxpool1 -> Conv2d_3b_1x1 -> Conv2d_4a_3x3 -> maxpool2 -> Inception Modules ...
    """
        children_to_fine_tune = {'first_conv': list(self.model.children())[0:1],  # Conv2d_1a_3x3
                                 'first_block': list(self.model.children())[0:4],  # include up to maxpool1
                                 'before_inception_modules': list(self.model.children())[0:7],  # ... maxpool2
                                 'first_inception_module': list(self.model.children())[0:8],  # ... 1st inception module
                                 }
        
        params_to_fine_tune = []
        for child in children_to_fine_tune[self.mode]:
            params_to_fine_tune += [param for param in child.parameters()]

        optimizer = torch.optim.Adam(params_to_fine_tune, lr=1e-4)  # match DDI experiments learning rate


"""
**Surgical Fine-tuning Notes**

**Layers.** 
We use the following naming convention for the layers of ResNet-26:
* First: Only the first conv layer.
* First 2 layers: The first conv layer of the entire network, and the first conv layer within the first block.
* First 2 blocks: The first conv layer, and the first block. 
* Last: The last fully-connected (FC) layer.

For RVT∗-small:
* First layer: First conv layer inside the first transformer block. 
* First block: First transformer block.
* Last: Head or the final fully connected layer.
"""
