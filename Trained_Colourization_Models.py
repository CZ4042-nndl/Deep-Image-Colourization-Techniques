import os
import time
import numpy as np 
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, root_dir, process_type):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir)]
        self.file_index_dict = {self.files[i]: i for i in range(len(self.files))}
        self.process_type = process_type
        print('File[0]:',self.files[0],'| Total Files:', len(self.files), '| Process:',self.process_type)

    def __len__(self):
         return len(self.files)

    def __getitem__(self, index):
        try:
            #*** Read the image from file ***
            self.rgb_img = cv2.imread(os.path.join(self.root_dir,self.files[index])).astype(np.float32) 
            self.rgb_img /= 255.0 
            
            #*** Resize the color image to pass to encoder ***
            rgb_encoder_img = cv2.resize(self.rgb_img, (224, 224))
            
            #*** Resize the color image to pass to decoder ***
            rgb_inception_img = cv2.resize(self.rgb_img, (300, 300))
            
            ''' Encoder Images '''
            #*** Convert the encoder color image to normalized lab space ***
            self.lab_encoder_img = cv2.cvtColor(rgb_encoder_img,cv2.COLOR_BGR2Lab) 
            
            #*** Splitting the lab images into l-channel, a-channel, b-channel ***
            l_encoder_img, a_encoder_img, b_encoder_img = self.lab_encoder_img[:,:,0],self.lab_encoder_img[:,:,1],self.lab_encoder_img[:,:,2]
            
            #*** Normalizing l-channel between [-1,1] ***
            l_encoder_img = l_encoder_img/50.0 - 1.0
            
            #*** Repeat the l-channel to 3 dimensions ***
            l_encoder_img = torchvision.transforms.ToTensor()(l_encoder_img)
            l_encoder_img = l_encoder_img.expand(3,-1,-1)
            
            #*** Normalize a and b channels and concatenate ***
            a_encoder_img = (a_encoder_img/128.0)
            b_encoder_img = (b_encoder_img/128.0)
            a_encoder_img = torch.stack([torch.Tensor(a_encoder_img)])
            b_encoder_img = torch.stack([torch.Tensor(b_encoder_img)])
            ab_encoder_img = torch.cat([a_encoder_img, b_encoder_img], dim=0)
            
            ''' Inception Images '''
            #*** Convert the inception color image to lab space ***
            self.lab_inception_img = cv2.cvtColor(rgb_inception_img,cv2.COLOR_BGR2Lab)
            
            #*** Extract the l-channel of inception lab image *** 
            l_inception_img = self.lab_inception_img[:,:,0]/50.0 - 1.0
             
            #*** Convert the inception l-image to torch Tensor and stack it in 3 channels ***
            l_inception_img = torchvision.transforms.ToTensor()(l_inception_img)
            l_inception_img = l_inception_img.expand(3,-1,-1)
            
            ''' return images to data-loader '''
            rgb_encoder_img = torchvision.transforms.ToTensor()(rgb_encoder_img)
            return l_encoder_img, ab_encoder_img, l_inception_img, rgb_encoder_img, self.files[index]
        
        except Exception as e:
            print('Exception at ',self.files[index], e)
            return torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), 'Error'

    def show_rgb(self, index):
        self.__getitem__(index)
        print("RGB image size:", self.rgb_img.shape)        
        cv2.imshow(self.rgb_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_lab_encoder(self, index):
        self.__getitem__(index)
        print("Encoder Lab image size:", self.lab_encoder_img.shape)
        cv2.imshow(self.lab_encoder_img)
        c2.waitKey(0)
        cv2.destroyAllWindows()

    def show_lab_inception(self, index):
        self.__getitem__(index)
        print("Inception Lab image size:", self.lab_inception_img.shape)
        cv2.imshow(self.lab_inception_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def show_other_images(self, index):
        a,b,c,d,_ = self.__getitem__(index)
        print("Encoder l channel image size:",a.shape)
        cv2.imshow((a.detach().numpy().transpose(1,2,0)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Encoder ab channel image size:",b.shape)
        cv2.imshow((b.detach().numpy().transpose(1,2,0)[:,:,0]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow((b.detach().numpy().transpose(1,2,0)[:,:,1]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Inception l channel image size:",c.shape)
        cv2.imshow(c.detach().numpy().transpose(1,2,0))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Color resized image size:",d.shape)
        cv2.imshow(d.detach().numpy().transpose(1,2,0))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class Testing_Image():
    def __init__(self, dataset: CustomDataset, index: int = None, filename: str = None):
        """
        Either Index or filename must be specified
        file must be in dataset to work
        """
        if index is not None and filename is not None:
            raise Exception("Either specify index or filename not both")
        elif index is None and filename is None:
            raise Exception("Please either specify index or filename, both found to be None")
        if filename is not None:
            index = dataset.file_index_dict.get(filename, -1)
            if index == -1:
                raise Exception(f"File {filename} Not Found in Root Dir {dataset.root_dir}")
        
        l_encoder_img, ab_encoder_img, l_inception_img, rgb_encoder_img, filename = dataset[index]
        self.encoder_img = l_encoder_img
        self.inception_img = l_inception_img
        
        self.path = os.path.join(dataset.root_dir, filename)
    
    def get_rgb(self, resize=True):
        img = cv2.imread(self.path)
        if resize:
            img = cv2.resize(img, (224, 224))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def get_gray(self, resize=True):
        img = cv2.imread(self.path)
        if resize:
            img = cv2.resize(img, (224, 224))
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def get_lab(self, only_gray_layer=False, resize=True):
        img = cv2.imread(self.path)
        if resize:
            img = cv2.resize(img, (224, 224))
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return lab_img if not only_gray_layer else lab_img[:, :, 0]
    
    def get_encoder_img(self, resize=True, direct_input=False):
        """
        Gets image to be fed into encoder of colourization model
        resize: if True will resize the image to (224, 224) which is the trained resolution
        direct_input: True If this image will directly be fed into the model
        """
        l_encoder_img = self.encoder_img

        return l_encoder_img if not direct_input else l_encoder_img.unsqueeze(0)
    
    def get_inception_img(self, direct_input=False):
        """
        Gets image to be fed into inception model
        direct_input: True If this image will directly be fed into the model
        """
        l_inception_img = self.inception_img

        return l_inception_img if not direct_input else l_inception_img.unsqueeze(0)

# Now we will load several different model architectures

# 1) Default Architecture (Starting point of our experiments
class Encoder_Default(nn.Module):
    def __init__(self):
        super(Encoder_Default,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
            
        )

    def forward(self, x):
        self.model = self.model.float()
        return self.model(x.float())


class FusionLayer_Default(nn.Module):
    def __init__(self):
        super(FusionLayer_Default,self).__init__()

    def forward(self, inputs, mask=None):
        ip, emb = inputs
        emb = torch.stack([torch.stack([emb],dim=2)],dim=3)
        emb = emb.repeat(1,1,ip.shape[2],ip.shape[3])
        fusion = torch.cat((ip,emb),1)
        return fusion


class Decoder_Default(nn.Module):
    def __init__(self, input_depth):
        super(Decoder_Default,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_depth, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0),     
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Colorization_Default(nn.Module):
    def __init__(self, depth_after_fusion=256):
        super(Colorization_Default,self).__init__()
        self.encoder = Encoder_Default()
        self.fusion = FusionLayer_Default()
        self.after_fusion = nn.Conv2d(in_channels=1256, out_channels=depth_after_fusion,kernel_size=1, stride=1,padding=0)
        self.bnorm = nn.BatchNorm2d(256)
        self.decoder = Decoder_Default(depth_after_fusion)

    def forward(self, img_l, img_emb):
        img_enc = self.encoder(img_l)
        # new_img_emb = torch.zeros_like(img_emb)
        fusion = self.fusion([img_enc, img_emb])
        fusion = self.after_fusion(fusion)
        fusion = self.bnorm(fusion)
        return self.decoder(fusion)


class Encoder_SESD(nn.Module):
    def __init__(self):
        super(Encoder_SESD,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
            
        )

    def forward(self, x):
        self.model = self.model.float()
        return self.model(x.float())


class FusionLayer_SESD(nn.Module):
    def __init__(self):
        super(FusionLayer_SESD,self).__init__()

    def forward(self, inputs, mask=None):
        ip, emb = inputs
        emb = torch.stack([torch.stack([emb],dim=2)],dim=3)
        emb = emb.repeat(1,1,ip.shape[2],ip.shape[3])
        fusion = torch.cat((ip,emb),1)
        return fusion


class Decoder_SESD(nn.Module):
    def __init__(self, input_depth):
        super(Decoder_SESD,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_depth, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0),


            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0),


            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0),
            
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0),     
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Colorization_SESD(nn.Module):
    def __init__(self, depth_after_fusion):
        super(Colorization_SESD,self).__init__()
        self.encoder = Encoder_SESD()
        self.fusion = FusionLayer_SESD()
        self.after_fusion = nn.Conv2d(in_channels=1256, out_channels=depth_after_fusion,kernel_size=1, stride=1,padding=0)
        self.bnorm = nn.BatchNorm2d(256)
        self.decoder = Decoder_SESD(depth_after_fusion)

    def forward(self, img_l, img_emb):
        img_enc = self.encoder(img_l)
        # new_img_emb = torch.zeros_like(img_emb)
        fusion = self.fusion([img_enc, img_emb])
        fusion = self.after_fusion(fusion)
        fusion = self.bnorm(fusion)
        return self.decoder(fusion)


class Encoder_NoFusion(nn.Module):
    def __init__(self):
        super(Encoder_NoFusion,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        self.model = self.model.float()
        return self.model(x.float())


class Decoder_NoFusion(nn.Module):
    def __init__(self, input_depth):
        super(Decoder_NoFusion,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_depth, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0),     
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Colorization_NoFusion(nn.Module):
    def __init__(self, depth_after_fusion):
        super(Colorization_NoFusion,self).__init__()
        self.encoder = Encoder_NoFusion()
        self.after_fusion = nn.Conv2d(in_channels=512, out_channels=depth_after_fusion,kernel_size=1, stride=1,padding=0)
        self.bnorm = nn.BatchNorm2d(depth_after_fusion)
        self.decoder = Decoder_NoFusion(depth_after_fusion)

    def forward(self, img_l):
        img_enc = self.encoder(img_l)
        fusion = self.after_fusion(img_enc)
        fusion = self.bnorm(fusion)
        return self.decoder(fusion)
    

class Encoder_Tiniest(nn.Module):
    def __init__(self):
        super(Encoder_Tiniest,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        self.model = self.model.float()
        return self.model(x.float())


class Decoder_Tiniest(nn.Module):
    def __init__(self):
        super(Decoder_Tiniest,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0),
            
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0),     
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Colorization_Tiniest(nn.Module):
    def __init__(self, depth_after_fusion):
        super(Colorization_Tiniest,self).__init__()
        self.encoder = Encoder_Tiniest()
        # self.after_fusion = nn.Conv2d(in_channels=512, out_channels=depth_after_fusion,kernel_size=1, stride=1,padding=0)
        # self.bnorm = nn.BatchNorm2d(depth_after_fusion)
        self.decoder = Decoder_Tiniest()

    def forward(self, img_l):
        img_enc = self.encoder(img_l)
        # fusion = self.after_fusion(img_enc)
        # fusion = self.bnorm(fusion)
        return self.decoder(img_enc)


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class Colorization_ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        self.lab_adjustment = nn.Tanh()  # Added nn.Tanh to the regular architecture to better match the lab format (ab output)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.lab_adjustment(self.conv_last(x))  # Added nn.Tanh to the regular architecture to better match the lab format (ab output)

        return out

class Decoder_RGB(nn.Module):
    def __init__(self, input_depth):
        super(Decoder_RGB,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_depth, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0),


            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0),


            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),     
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
class Colorization_RGB(nn.Module):
    def __init__(self, depth_after_fusion):
        super(Colorization_RGB,self).__init__()
        self.encoder = Encoder_Default()
        self.fusion = FusionLayer_Default()
        self.after_fusion = nn.Conv2d(in_channels=1256, out_channels=depth_after_fusion,kernel_size=1, stride=1,padding=0)
        self.bnorm = nn.BatchNorm2d(256)
        self.decoder = Decoder_RGB(depth_after_fusion)

    def forward(self, img_l, img_emb):
        img_enc = self.encoder(img_l)
        # new_img_emb = torch.zeros_like(img_emb)
        fusion = self.fusion([img_enc, img_emb])
        fusion = self.after_fusion(fusion)
        fusion = self.bnorm(fusion)
        return self.decoder(fusion)

class Decoder_LAB(nn.Module):
    def __init__(self, input_depth):
        super(Decoder_LAB,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_depth, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0),


            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0),


            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),    
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Colorization_LAB(nn.Module):
    def __init__(self, depth_after_fusion):
        super(Colorization_LAB,self).__init__()
        self.encoder = Encoder_Default()
        self.fusion = FusionLayer_Default()
        self.after_fusion = nn.Conv2d(in_channels=1256, out_channels=depth_after_fusion,kernel_size=1, stride=1,padding=0)
        self.bnorm = nn.BatchNorm2d(256)
        self.decoder = Decoder_LAB(depth_after_fusion)

    def forward(self, img_l, img_emb):
        img_enc = self.encoder(img_l)
        # new_img_emb = torch.zeros_like(img_emb)
        fusion = self.fusion([img_enc, img_emb])
        fusion = self.after_fusion(fusion)
        fusion = self.bnorm(fusion)
        return self.decoder(fusion)
    

# Now we load different model runners (they will make it easier to run and test a model and handle all the pre and post processing)

class Base_Model_Runner():
    """
    Base Class for all model runners
    Hold all common methods
    """
    def get_model(self):
        return self.model
    
    def _concatente_and_colorize(self, im_lab, img_ab):
        # Assumption is that im_lab is of size [1,1,224,224]
        # print(im_lab.size(),img_ab.size())
        np_img = im_lab[0].cpu().detach().numpy().transpose(1,2,0)
        lab = np.empty([*np_img.shape[0:2], 3],dtype=np.float32)
        lab[:, :, 0] = np.squeeze(((np_img + 1) * 50))
        lab[:, :, 1:] = img_ab[0].cpu().detach().numpy().transpose(1,2,0) * 127
        np_img = cv2.cvtColor(lab,cv2.COLOR_Lab2RGB) 
        color_im = torch.stack([torchvision.transforms.ToTensor()(np_img)],dim=0)
        # color_img_jpg = color_im[0].detach().numpy().transpose(1,2,0)
        return color_im
    
    def get_image_output(self, input_image: Testing_Image) -> np.ndarray:
        inception_img = input_image.get_inception_img(direct_input=False)
        img_embs = self.inception_model(inception_img.float().unsqueeze(0))
        output_ab = self.model(input_image.get_encoder_img(direct_input=False).unsqueeze(0), img_embs)
        new_lab_image = torch.stack([input_image.get_encoder_img(direct_input=True)[:,0,:,:]], dim=1)
        color_img = self._concatente_and_colorize(new_lab_image, output_ab)
        color_img_jpg = color_img[0].detach().numpy().transpose(1,2,0)
        return color_img_jpg
    
    def _get_image_output_no_fusion(self, input_image: Testing_Image) -> np.ndarray:
        output_ab = self.model(input_image.get_encoder_img(direct_input=False).unsqueeze(0))
        new_lab_image = torch.stack([input_image.get_encoder_img(direct_input=True)[:,0,:,:]], dim=1)
        color_img = self._concatente_and_colorize(new_lab_image, output_ab)
        color_img_jpg = color_img[0].detach().numpy().transpose(1,2,0)
        return color_img_jpg

class Default_Model_Runner(Base_Model_Runner):
    def __init__(self, checkpoint_path="Models/checkpoint18.pt"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = Colorization_Default(256).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()

        self.inception_model = models.inception_v3(pretrained=True).float().to(device)
        self.inception_model = self.inception_model.float()
        self.inception_model.eval()

class SESD_Model_Runner(Base_Model_Runner):
    def __init__(self, checkpoint_path="Models/sesd_checkpoint19.pt"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = Colorization_SESD(256).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()

        self.inception_model = models.inception_v3(pretrained=True).float().to(device)
        self.inception_model = self.inception_model.float()
        self.inception_model.eval()

class NoFusion_Model_Runner(Base_Model_Runner):
    def __init__(self, checkpoint_path="Models/no_fusion_checkpoint19.pt"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = Colorization_NoFusion(256).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
    
    def get_image_output(self, input_image: Testing_Image) -> np.ndarray:
        return self._get_image_output_no_fusion(input_image)
    
class Tiniest_Model_Runner(Base_Model_Runner):
    def __init__(self, checkpoint_path="Models/tiniest_checkpoint19.pt"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = Colorization_Tiniest(256).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
    
    def get_image_output(self, input_image: Testing_Image) -> np.ndarray:
        return self._get_image_output_no_fusion(input_image)
    
class ResNetUNet_Model_Runner(Base_Model_Runner):
    def __init__(self, checkpoint_path="Models/ResNetUNet_checkpoint19.pt") -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = Colorization_ResNetUNet(2).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
    
    def get_image_output(self, input_image: Testing_Image) -> np.ndarray:
        return self._get_image_output_no_fusion(input_image)

class RGB_Model_Runner(Base_Model_Runner):
    """Model_Runner class for Default Architecture Model returning RGB Channels instead of just AB part of LAB"""
    def __init__(self, checkpoint_path="Models/RGBcheckpoint19.pt"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = Colorization_RGB(256).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()

        self.inception_model = models.inception_v3(pretrained=True).float().to(device)
        self.inception_model = self.inception_model.float()
        self.inception_model.eval()

    def _process_RGB_output(self, output_rgb):
        rgb_channels = output_rgb[0].cpu().detach().numpy().transpose(1,2,0) # (224,224,3)
        im_rgb_processed = (rgb_channels * 255).astype(np.uint8)
        im_rgb_processed = cv2.cvtColor(im_rgb_processed,cv2.COLOR_BGR2RGB)
        im_rgb_processed = torchvision.transforms.ToTensor()(im_rgb_processed) # (3,224,224) # auto normalise under the hood, dont need to * 255 again
        return im_rgb_processed
    
    def get_image_output(self, input_image: Testing_Image) -> np.ndarray:
        inception_img = input_image.get_inception_img(direct_input=False)
        img_embs = self.inception_model(inception_img.float().unsqueeze(0))
        output_rgb = self.model(input_image.get_encoder_img(direct_input=False).unsqueeze(0), img_embs)
        output_rgb_processed = self._process_RGB_output(output_rgb)
        # save_image(output_rgb_processed,'./Outputs/'+file_name[0])
        output_rgb_img = output_rgb_processed.cpu().detach().numpy()
        color_img_jpg = output_rgb_img.transpose(1,2,0)
        return color_img_jpg

class LAB_Model_Runner(Base_Model_Runner):
    """Model_Runner class for Default Architecture Model returning all LAB Channels instead of just AB"""
    def __init__(self, checkpoint_path="Models/LABcheckpoint19.pt"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = Colorization_LAB(256).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()

        self.inception_model = models.inception_v3(pretrained=True).float().to(device)
        self.inception_model = self.inception_model.float()
        self.inception_model.eval()
    
    def _process_LAB_output(self, im_lab): # A,B,L
        lab_channels = im_lab[0].cpu().detach().numpy().transpose(1,2,0) # this transpose is to form (height, width, channel)
        l, ab = lab_channels[:, :, 2], lab_channels[:, :, :2] # A,B,L
        lab = np.empty([*lab_channels.shape[0:2], 3],dtype=np.float32)
        lab[:, :, 0] = (l + 1.0) * 50.0
        lab[:, :, 1:] = ab * 127.0
        np_img = cv2.cvtColor(lab,cv2.COLOR_Lab2RGB) 
        color_im = torch.stack([torchvision.transforms.ToTensor()(np_img)],dim=0)
        return color_im
    
    def get_image_output(self, input_image: Testing_Image) -> np.ndarray:
        inception_img = input_image.get_inception_img(direct_input=False)
        img_embs = self.inception_model(inception_img.float().unsqueeze(0))
        output_lab = self.model(input_image.get_encoder_img(direct_input=False).unsqueeze(0), img_embs)
        output_rgb = self._process_LAB_output(output_lab)
        output_rgb_img = output_rgb[0].detach().numpy().transpose(1,2,0)

        return output_rgb_img

