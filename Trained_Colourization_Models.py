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
        output_ab = self.model(input_image.get_encoder_img(direct_input=False).unsqueeze(0))
        new_lab_image = torch.stack([input_image.get_encoder_img(direct_input=True)[:,0,:,:]], dim=1)
        color_img = self._concatente_and_colorize(new_lab_image, output_ab)
        color_img_jpg = color_img[0].detach().numpy().transpose(1,2,0)
        return color_img_jpg

