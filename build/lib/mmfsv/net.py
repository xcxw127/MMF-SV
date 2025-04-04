import math
import os
import random
from multiprocessing import Pool, cpu_count

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions.multivariate_normal as mn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pudb import set_trace
from pytorch_lightning.core.hooks import CheckpointHooks
from pytorch_msssim import MS_SSIM
from ray import tune
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from transformers import AlbertModel
import clip
from PIL import Image
from mmfsv import utilities as ut


class MultiLP(nn.Module):

    def __init__(self, full_dim):

        super(MultiLP, self).__init__()
        dim1 = full_dim[:-1]  # Input dimensions for each layer
        dim2 = full_dim[1:]   # Output dimensions for each layer

        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(k, m),
                nn.ReLU(),
            ) for k, m in zip(dim1, dim2)
        )

    def forward(self, x):

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
        return out


class classfication(nn.Module):

    def __init__(self, full_dim):

        super(classfication, self).__init__()
        dim1 = full_dim[:-1]
        dim2 = full_dim[1:]
        # Create a list of sequential layers (Linear + ReLU)
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(k, m),
                nn.ReLU(),
            ) for k, m in zip(dim1, dim2)
        )

    def forward(self, x):

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
        return out


class attention(nn.Module):

    def __init__(self, dim, out_dim):

        super(attention, self).__init__()
        # Sequential layers for Query and Key
        self.Q_K = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.Sigmoid(),
        )
        # Sequential layers for Value
        self.V = nn.Sequential(
            nn.Linear(dim, out_dim),
        )

    def forward(self, x):

        qk = self.Q_K(x)  # Compute Query and Key
        v = self.V(x)     # Compute Value
        out = torch.mul(qk, v)  # Element-wise multiplication
        return out


class attention_classfication(nn.Module):

    def __init__(self, full_dim):

        super(attention_classfication, self).__init__()
        dim1 = full_dim[:-1]
        dim2 = full_dim[1:]
        # Create a list of sequential layers including attention, Linear, and ReLU
        self.layers = nn.ModuleList(
            nn.Sequential(
                attention(k, m),
                nn.Linear(m, m),
                nn.ReLU(inplace=True),
            ) for k, m in zip(dim1, dim2)
        )

    def forward(self, x):

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
        return out


class resnet_attention_classfication(nn.Module):

    def __init__(self, full_dim):

        super(resnet_attention_classfication, self).__init__()
        dim1 = full_dim[:-1]
        dim2 = full_dim[1:]
        # Create attention-based layers
        self.layers = nn.ModuleList(
            nn.Sequential(
                attention(k, m),
                nn.Linear(m, m),
                nn.ReLU(inplace=True),
            ) for k, m in zip(dim1, dim2)
        )

        # Create residual layers for even-indexed layers
        self.res2 = nn.ModuleList(
            nn.Sequential(
                nn.Linear(full_dim[2 * index], full_dim[2 * (index + 1)]),
                nn.ReLU(inplace=True),
            ) for index in range(int((len(full_dim) - 1) / 2))
        )

    def forward(self, x):

        out = x
        for i in range(len(self.layers)):
            if i % 2 == 0:
                # For even-indexed layers, store the current output for residual connection
                x = out
                out = self.layers[i](out)
            else:
                # For odd-indexed layers, add residual connection
                out = self.layers[i](out) + self.res2[int(i / 2)](x)

        return out


class conv2ds_sequential(nn.Module):

    def __init__(self, full_dim):

        super(conv2ds_sequential, self).__init__()
        dim1 = full_dim[:-1]
        dim2 = full_dim[1:]
        # Create a list of sequential Conv2D, BatchNorm, and ReLU layers
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=k, out_channels=m, kernel_size=3,
                          stride=1, padding=1),  # Maintain spatial dimensions
                nn.BatchNorm2d(m),
                nn.ReLU(inplace=True),
            ) for k, m in zip(dim1, dim2)
        )

    def forward(self, x):

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
        return out


class deconv2ds_sequential(nn.Module):

    def __init__(self, full_dim):

        super(deconv2ds_sequential, self).__init__()
        dim1 = full_dim[:-1]
        dim2 = full_dim[1:]
        # Reverse the dimensions for upsampling
        dim1.reverse()
        dim2.reverse()
        # Create a list of sequential ConvTranspose2D, BatchNorm, and ReLU layers
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=m, out_channels=k,
                                   kernel_size=3, stride=1, padding=1),  # Maintain spatial dimensions
                nn.BatchNorm2d(k),
                nn.ReLU(inplace=True),
            ) for k, m in zip(dim1, dim2)
        )
        # Final activation
        self.tanh = nn.Tanh()

    def forward(self, x):

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
        out = self.tanh(out)
        return out


class conv2ds_after_resnet(nn.Module):

    def __init__(self, in_dim, out_dim):

        super(conv2ds_after_resnet, self).__init__()
        # First set of Conv2D layers
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=k, out_channels=k+1,
                          kernel_size=3, stride=1),  # Increase channel by 1
                nn.BatchNorm2d(k+1),
                nn.Conv2d(in_channels=k+1, out_channels=k+1,
                          kernel_size=3, stride=1),  # Maintain channel
                nn.ReLU(inplace=True),
            ) for k in range(in_dim, out_dim)
        )

        # Second set of Conv2D layers for residual connection
        self.layers2 = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=k, out_channels=k+1,
                          kernel_size=3, stride=1),  # Increase channel by 1
                nn.BatchNorm2d(k+1),
                nn.ReLU(inplace=True),
            ) for k in range(in_dim, out_dim)
        )

    def forward(self, x):

        out = x
        for i in range(len(self.layers)):
            # Add outputs of both Conv2D paths
            out = self.layers[i](out) + self.layers2[i](out)

        return out


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, use_sigmoid=True):

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):

        if self.use_sigmoid:
            pred = self.sigmoid(pred)  # Apply sigmoid if specified
        pred = pred.view(-1)
        label = target.view(-1)
        pos = torch.nonzero(label > 0).squeeze(1)  # Indices of positive samples
        pos_num = max(pos.numel(), 1.0)  # Number of positive samples
        mask = ~(label == -1)  # Mask to exclude labels with -1
        pred = pred[mask]
        label = label[mask]
        # Compute focal weights
        focal_weight = self.alpha * (label - pred).abs().pow(self.gamma) * (label > 0.0).float(
        ) + (1 - self.alpha) * pred.abs().pow(self.gamma) * (label <= 0.0).float()
        # Compute binary cross-entropy loss
        loss = F.binary_cross_entropy(
            pred, label, reduction='none') * focal_weight
        return loss.sum() / pos_num  # Normalize by number of positive samples


class IDENet(pl.LightningModule):

    def __init__(self, path, config, target_seq_len=77):

        super(IDENet, self).__init__()

        self.train_outputs = []
        self.validation_outputs = []
        self.test_outputs = []

        # Hyperparameters and settings from config
        self.lr = config["lr"]
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        self.weight_decay = config['weight_decay']
        self.batch_size = config["batch_size"]
        self.model_name = config["model_name"]
        self.use_kfold = config.get("use_kfold", False)
        if self.use_kfold:
            self.KFold = config["KFold"]
            self.data_index = config["KFold_num"]
        self.data_dir = path

        # Define dimensions for convolutional layers
        conv2d_dim = list(range(7, 3, -1))  # [7, 6, 5, 4]
        conv2d_dim.append(3)  # Append 3 to make [7,6,5,4,3]
        self.conv2ds = conv2ds_sequential(conv2d_dim)  # Sequential Conv2D layers
        self.deconv2ds = deconv2ds_sequential(conv2d_dim)  # Sequential DeConv2D layers

        # Define dimensions for classification layers
        # full_dim = [1000 + 768, 768 * 2, 768, 384, 192, 96, 48, 24, 12, 6]
        full_dim = [1024, 768, 384, 192, 96, 48, 24, 12, 6]
        # full_dim = [512, 384, 192, 96, 48, 24, 12, 6]
        self.classfication = resnet_attention_classfication(full_dim)  # Classification network with attention and ResNet

        # self.classifier = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(512, 3),
        #     nn.Softmax(dim=1)
        # )

        # Define softmax layer for final classification
        self.softmax = nn.Sequential(
            nn.Linear(full_dim[-1], 3),
            nn.Softmax(1)
        )

        # Define loss functions and modules
        self.criterion = FocalLoss()  # Focal loss for class imbalance
        self.ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=conv2d_dim[0])  # MS-SSIM for image similarity
        self.L1 = nn.SmoothL1Loss(reduction='mean')  # L1 loss for reconstruction

        # # Initialize ResNet model based on the specified model name
        # self.resnet_model = eval("torchvision.models." + self.model_name)(pretrained=True)  # Pretrained ResNet

        # # Initialize BERT model
        # self.bert = torch.load("/home/xzy/Desktop/F-SV/models/init_albert.pt")  # Load pre-trained BERT
        # self.bert.eval()  # Set BERT to evaluation mode
        # # Modify BERT embeddings to custom dimensions
        # self.bert.embeddings.word_embeddings = nn.modules.sparse.Embedding(30000, 11, padding_idx=0)
        # self.bert.embeddings.position_embeddings = nn.modules.sparse.Embedding(512, 11)
        # self.bert.embeddings.token_type_embeddings = nn.modules.sparse.Embedding(2, 11)
        # self.bert.embeddings.LayerNorm = nn.modules.normalization.LayerNorm((11,), eps=1e-12, elementwise_affine=True)
        # self.bert.encoder.embedding_hidden_mapping_in = nn.modules.linear.Linear(in_features=11, out_features=768, bias=True)

        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False  # freeze CLIP parameters
        
        self.target_seq_len = target_seq_len

    def forward(self, x1, x2):

        # Pass image through convolutional layers
        x_feature = self.conv2ds(x1)

        # Reconstruct image using deconvolutional layers
        re_x1 = self.deconv2ds(x_feature)
        
        # Compute reconstruction loss using MS-SSIM and L1 loss
        reconstruct_loss = 1 - self.ms_ssim_module(x1, re_x1) + self.L1(x1, re_x1) / 255

        # # ResNet + BERT method
        # # Pass features through ResNet model
        # x1 = self.resnet_model(x_feature)
        # # Pass text through BERT model and get the pooled output
        # x2 = self.bert(inputs_embeds=x2)[1]

        # # Concatenate ResNet and BERT outputs and pass through classification network
        # y_hat = self.classfication(torch.cat([x1, x2], 1))
        # y_hat = self.softmax(y_hat)  # Apply softmax for probabilities

        # CLIP method
        x2 = x2.long().to(self.device)

        # Pad or cut x2 size to 77
        if x2.size(1) < self.target_seq_len:
            padding = (0, self.target_seq_len - x2.size(1))  # pad to target sequence length
            x2 = torch.nn.functional.pad(x2, padding)
        elif x2.size(1) > self.target_seq_len:
            x2 = x2[:, :self.target_seq_len, :] # cut

        # Use CLIP for image and text tarits
        with torch.no_grad():
            # Tunnels should be 3
            clip_image_features = self.clip_model.encode_image(x_feature)  # output size: [batch_size, 512]
            clip_text_features = self.clip_model.encode_text(x2)    # output size: [batch_size, 512]
            # normalization the traits
            clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
            clip_text_features /= clip_text_features.norm(dim=-1, keepdim=True)

        # Combine CLIP features
        combined_features = torch.cat([clip_image_features, clip_text_features], dim=1)

        # Classification model predict
        y_hat = self.classfication(combined_features)
        # y_hat = self.classfication(clip_image_features)
        y_hat = self.softmax(y_hat)

        return y_hat

    def training_validation_step(self, batch, batch_idx):

        x, y = batch  # x2(length, 12)
        del batch  # Free up memory
        x1 = x["image"]  # Image input
        x2 = x["list"]   # Text input (e.g., BERT embeddings)

        # copy_image = x1[:, 6:7, :, :]  # size [1280, 1, 224, 224]
        # x1_modified = copy_image.repeat(1, 7, 1, 1)  # size [1280, 7, 224, 224]

        # Pass image through convolutional layers
        x_feature = self.conv2ds(x1)
        # x_feature = self.conv2ds(x1_modified)

        # Reconstruct image using deconvolutional layers
        re_x1 = self.deconv2ds(x_feature)

        # Compute reconstruction loss
        reconstruct_loss = 1 - self.ms_ssim_module(x1, re_x1) + self.L1(x1, re_x1) / 255


        # # ResNet + BERT method
        # # Pass features through ResNet model
        # x1 = self.resnet_model(x_feature)
        # # Pass text through BERT model and get the pooled output
        # x2 = self.bert(inputs_embeds=x2)[1]

        # # Initialize one-hot encoded target tensor
        # y_t = torch.empty(len(y), 3).cuda()
        # for i, y_item in enumerate(y):
        #     if y_item == 0:
        #         y_t[i] = torch.tensor([1, 0, 0])
        #     elif y_item == 1:
        #         y_t[i] = torch.tensor([0, 1, 0])
        #     else:
        #         y_t[i] = torch.tensor([0, 0, 1])

        # # Concatenate ResNet and BERT outputs and pass through classification network
        # y_hat = self.classfication(torch.cat([x1, x2], 1))
        # y_hat = self.softmax(y_hat)  # Apply softmax for probabilities

        # # Compute total loss as sum of focal loss and weighted reconstruction loss
        # loss = self.criterion(y_hat, y_t) + 0.2 * reconstruct_loss


        # CLIP method
        x2 = x2.long().to(self.device)

        # # Use ResNet model for image traits
        # x1_resnet = self.resnet_model(x_feature)  # output size: [batch_size, 1000]
        # # Use BERT for text traits
        # x2_bert = self.bert(input_ids=x2)[1]

        # Pad or cut x2 size to 77
        if x2.size(1) < self.target_seq_len:
            padding = (0, self.target_seq_len - x2.size(1))  # pad to target sequence length
            x2 = torch.nn.functional.pad(x2, padding)
        elif x2.size(1) > self.target_seq_len:
            x2 = x2[:, :self.target_seq_len, :] # cut

        # Use CLIP for image and text tarits
        with torch.no_grad():
            # Tunnels should be 3
            clip_image_features = self.clip_model.encode_image(x_feature)  # output size: [batch_size, 512]
            clip_text_features = self.clip_model.encode_text(x2)    # output size: [batch_size, 512]

            # normalization the traits
            clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
            clip_text_features /= clip_text_features.norm(dim=-1, keepdim=True)

        # Combine CLIP features
        combined_features = torch.cat([clip_image_features, clip_text_features], dim=1)

        # # Classification model predict
        y_hat = self.classfication(combined_features)
        # y_hat = self.classfication(clip_image_features)
        y_hat = self.softmax(y_hat)

        # Transform labels to one-hot code
        y_t = torch.zeros(len(y), 3).to(y.device)
        y_t.scatter_(1, y.view(-1, 1), 1)

        # Calculate loss
        loss = self.criterion(y_hat, y_t) + 0.2 * reconstruct_loss

        return loss, y, y_hat


    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self.training_validation_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.train_outputs.append({'loss': loss.detach(), 'y': y.detach(), 'y_hat': torch.argmax(y_hat, dim=1).detach()})

        return loss 


    def on_train_epoch_end(self):
        y, y_hat = [], []
        for out in self.train_outputs:
            y.extend(out['y'].cpu())
            y_hat.extend(out['y_hat'].cpu())

        y = torch.tensor(y).reshape(-1)
        y_hat = torch.tensor(y_hat).reshape(-1)

        metric = classification_report(y, y_hat, output_dict=True)

        self.log('train_mean', metric['accuracy'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', metric['macro avg']['f1-score'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_macro_pre', metric['macro avg']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_macro_re', metric['macro avg']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_0_pre', metric['0']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_0_re', metric['0']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_1_pre', metric['1']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_1_re', metric['1']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_2_pre', metric['2']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_2_re', metric['2']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # clear
        self.train_outputs.clear()


    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self.training_validation_step(batch, batch_idx)
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.validation_outputs.append({'y': y.detach(), 'y_hat': torch.argmax(y_hat, dim=1).detach()})


    def on_validation_epoch_end(self):
        y, y_hat = [], []
        for out in self.validation_outputs:
            y.extend(out['y'].cpu())
            y_hat.extend(out['y_hat'].cpu())

        y = torch.tensor(y).reshape(-1)
        y_hat = torch.tensor(y_hat).reshape(-1)

        metric = classification_report(y, y_hat, output_dict=True)

        self.log('validation_mean', metric['accuracy'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_f1', metric['macro avg']['f1-score'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_macro_pre', metric['macro avg']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_macro_re', metric['macro avg']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_0_pre', metric['0']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_0_re', metric['0']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_1_pre', metric['1']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_1_re', metric['1']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_2_pre', metric['2']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_2_re', metric['2']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        tune.report(validation_f1=metric['macro avg']['f1-score'])

        # clear
        self.validation_outputs.clear()


    def test_step(self, batch, batch_idx):
        loss, y, y_hat = self.training_validation_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.test_outputs.append({'y': y.detach(), 'y_hat': y_hat.detach()})


    def on_test_epoch_end(self):
        y_all, y_hat_all = [], []
        for out in self.test_outputs:
            y_all.extend(out['y'].cpu())
            y_hat_all.extend(out['y_hat'].cpu())

        y_all = torch.tensor(y_all, dtype=torch.long).reshape(-1)
        y_hat_all = torch.stack(y_hat_all).reshape(-1, 3)

        torch.save({"y": y_all, "y_hat": y_hat_all}, "result.pt")

        metric = classification_report(y_all, torch.argmax(y_hat_all, dim=1), output_dict=True)

        self.log('test_mean', metric['accuracy'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', metric['macro avg']['f1-score'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_macro_pre', metric['macro avg']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_macro_re', metric['macro avg']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_0_pre', metric['0']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_0_re', metric['0']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_1_pre', metric['1']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_1_re', metric['1']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_2_pre', metric['2']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_2_re', metric['2']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        print("\n[TEST] Classification Report:\n", classification_report(y_all, torch.argmax(y_hat_all, dim=1), digits=4))

        # clear
        self.test_outputs.clear()


    def prepare_data(self):
        # Load data identifiers and labels
        x_ids, labels = ut.load_class(self.data_dir)

        if self.use_kfold:
            five_folds = StratifiedKFold(n_splits=self.KFold, shuffle=True, random_state=2022)
            for i, (train_idx1, test_idx1) in enumerate(five_folds.split(x_ids, labels)):
                if i == self.data_index:
                    train_idx, test_idx = train_idx1, test_idx1
                    break
        else:
            train_idx, test_idx = train_test_split(
                range(len(x_ids)),
                test_size=0.2,
                stratify=labels,
                random_state=2022
            )

        # Initialize the dataset
        input_data = ut.IdentifyDataset(self.data_dir)

        # Shuffle the indices for reproducibility
        random.seed(2022)
        random.shuffle(train_idx)
        random.shuffle(test_idx)

        # Create training and testing subsets
        self.train_dataset = Subset(input_data, train_idx)
        self.test_dataset = Subset(input_data, test_idx)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=int(cpu_count()), prefetch_factor=10, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size * 20,  pin_memory=True, num_workers=int(cpu_count()))

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size * 20, pin_memory=True, num_workers=int(cpu_count()))
    
    def configure_optimizers(self):
        # Use Adam optimizer with specified hyperparameters
        # opt_e = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        # Uncomment below to add a separate optimizer for another component (e.g., discriminator)
        # opt_d = torch.optim.Adam(self.line.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        opt_e = torch.optim.Adam(self.parameters(), lr=1e-4)
        return [opt_e]
