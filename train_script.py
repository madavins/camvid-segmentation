import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from unet import *
import metrics_sauc
import losses
import os
import numpy as np
import matplotlib.pyplot as plt
import dataset
import dataset_coco
from tqdm import tqdm
import albumentations as A
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from deeplab2.deeplabv3 import DeepLabV3Plus
from deeplab2 import encoders
from camvid_unet import *

#CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, plot=True):
    start = time.time()

    train_loss = []
    train_wiou = []
    train_oacc = []
    train_miou = []

    validation_loss = []
    validation_wiou = []
    validation_oacc = []
    validation_miou = []

    max_score = 0.0

    for epoch in range(epochs):
        #TRAIN
        model.train() 
        epoch_loss = 0
        epoch_wmiou = 0
        epoch_miou = 0
        epoch_acc = 0

        with tqdm(total=len(train_data), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch_idx, batch in enumerate(train_loader, 0):
                optimizer.zero_grad()
                images, true_masks = batch
                #images shape: [batch * 3 * 360 * 480]
                #true_masks shape: [batch * 360 * 480]
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                pred_masks = model(images)#pred_masks shape: [batch * Nclasses * 360 * 480]
                
                loss = criterion(pred_masks, true_masks)
                epoch_loss += loss.item()
                
                wmiou, miou, _   = mIoU(pred_masks, true_masks)
                epoch_wmiou += wmiou.cpu().detach().numpy()
                epoch_miou += miou.cpu().detach().numpy()
                acc, _ = accuracy(pred_masks, true_masks)
                epoch_acc += acc.cpu().detach().numpy()
                
                pbar.set_postfix(**{'Loss': epoch_loss / (batch_idx+1),
                                    'wIoU': epoch_wmiou / (batch_idx+1),
                                    'Accuracy': epoch_acc / (batch_idx+1),
                                    'mIoU': epoch_miou / (batch_idx+1)})
                pbar.update(images.shape[0])

                loss.backward()
                optimizer.step()

        if scheduler != None:
            #scheduler.step()
            scheduler.step(epoch_acc) #Reduce on plateau parameter maximitzation
        
        train_loss.append(epoch_loss / (batch_idx+1))
        train_wiou.append(epoch_wmiou / (batch_idx+1))
        train_oacc.append(epoch_acc / (batch_idx+1))
        train_miou.append(epoch_miou / (batch_idx+1))

        #VALIDATION
        model.eval()
        val_loss = 0
        val_wmiou = 0
        val_miou = 0
        val_acc = 0

        with tqdm(total=len(val_data), desc='Validation set', unit='img') as pbar:
            for batch_idx, batch in enumerate(val_loader, 0):
                images, true_masks = batch
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.no_grad():
                    pred_masks = model(images)

                val_loss += criterion(pred_masks, true_masks).item()
                
                wmiou, miou, _   = mIoU(pred_masks, true_masks)
                val_wmiou += wmiou.cpu().detach().numpy()
                acc, _ = accuracy(pred_masks, true_masks)
                val_acc += acc.cpu().detach().numpy()
                val_miou += miou.cpu().detach().numpy()

                pbar.set_postfix(**{'Loss': val_loss / (batch_idx+1),
                                   'wIoU': val_wmiou / (batch_idx+1),
                                   'Accuracy': val_acc / (batch_idx+1),
                                   'mIoU': val_miou / (batch_idx+1)})
                
                pbar.update(images.shape[0])

        validation_loss.append(val_loss / (batch_idx+1))
        validation_wiou.append(val_wmiou / (batch_idx+1))
        validation_oacc.append(val_acc / (batch_idx+1))
        validation_miou.append(val_miou / (batch_idx+1))
        
        if max_score < val_wmiou: #Save the best model (IoU)
            max_score = val_wmiou
            torch.save(model, './new_camvid/unet_patches2finetune_wce_adam_0001.pth')
            print('Model saved!')

    print('Finished Training!')
    stop = time.time()
    hh_mm_ss= time.strftime('%H:%M:%S', time.gmtime(stop-start))
    print(f"Training time: {hh_mm_ss}")

    #PLOTS
    if plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots( nrows=2, ncols=2,figsize=(15,10))  # create figure & 1 axis

        ax1.plot(range(0, epochs), train_loss, label="Training Loss")
        ax1.plot(range(0, epochs), validation_loss, label="Validation Loss")
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training vs Validation Loss')
        ax1.legend()
        ax2.plot(range(0, epochs), train_oacc, label="Training Accuracy")
        ax2.plot(range(0, epochs), validation_oacc, label="Validation Accuracy")
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Overall Accuracy')
        ax2.set_title('Training vs Validation Accuracy')
        ax2.legend()
        ax3.plot(range(0, epochs), train_wiou, label="Training wmIoU")
        ax3.plot(range(0, epochs), validation_wiou, label="Validation wmIoU")
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('wmIoU')
        ax3.set_title('Training vs Validation weighted mIoU')
        ax3.legend()
        ax4.plot(range(0, epochs), train_miou, label="Training mIoU")
        ax4.plot(range(0, epochs), validation_miou, label="Validation mIoU")
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('mIoU')
        ax4.set_title('Training vs Validation mIoU')
        ax4.legend()

        fig.savefig('./new_camvid/unet_patches2finetune_wce_adam_0001.png')

if __name__ == "__main__":
    #Load dataset
    DATA_DIR = './data/CamVid/'
    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')
    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'validannot')
    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')

    def transformation():  
        transform = [
            A.RandomCrop(width=480, height=352,  p=1),
            A.HorizontalFlip(),
        ]
        return A.Compose(transform)

    
    train_data = dataset.Dataset(x_train_dir, y_train_dir, augmentation = None)
    val_data = dataset.Dataset(x_valid_dir, y_valid_dir, augmentation = None)
    #train_data = dataset_coco.Dataset(x_train_dir, y_train_dir, augmentation = val_transformation())
    #val_data = dataset_coco.Dataset(x_valid_dir, y_valid_dir, augmentation = val_transformation())
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

    #Define my model:
    #DeeplabV3+
    #model = DeepLabV3Plus(encoder_name = "resnet101",
            #encoder_depth = 5,
            #encoder_weights = "imagenet",
            #encoder_output_stride = 16,
            #decoder_channels = 256,
            #decoder_atrous_rates = (6, 12, 18),
            #in_channels = 3,
            #classes = 32,
            #upsampling = 4)
    #DeeplabV3
    #model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=True)
    #U-Net
    model = CamvidUNet(3, 32)
    model = torch.load('/home/usuaris/imatge/manel.davins/new_camvid/unet_patches2_wce_adam_0001.pth');
    model.to(device);
    
    #Metrics:
    mIoU = metrics_sauc.Jaccard()
    accuracy = metrics_sauc.OAAcc()

    #Define optimizer
    lr = 1e-4 / 5
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=0.0005)
    #optimizer = optim.SGD(model.parameters(), lr, momentum=0.99)
    #cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr = 1e-3)
    
    #Define Loss
    class_weights = torch.FloatTensor([50.37676985, 49.31400563, 40.6869888 , 49.4280702 ,  4.39475015,
       19.65531343, 49.83795652, 49.70221303, 33.07967799, 27.33550434,
       25.94484908, 50.21071836, 37.84765877, 50.35634767, 40.74921503,
       42.80685005, 38.38857375,  3.65530721, 44.91934535, 11.54580361,
       47.35963121,  6.16453288, 39.77155587, 50.39863981, 42.61361366,
       50.49834979,  8.52220088, 46.10246534, 50.49830932, 35.6327177 ,
       21.73201866, 31.0631698 ]).cuda()
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    boundary_loss = losses.BoundaryLoss()
    dice_loss = losses.DiceLoss()
    focal_loss = losses.FocalLoss(alpha=class_weights)

    #Define scheduler
    step_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=140, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, verbose=True)

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100, plot=True)

    
    