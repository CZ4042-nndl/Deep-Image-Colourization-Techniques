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
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class Configuration:
    # TODO: create Models/ folder and ensure path is correct
    data_path = "../../nndl-project-data/coco2017" #TODO: change your data path, must end with /coco2017
    model_file_name = 'checkpoint19.pt' #TODO: change your file name if load_model_to_train OR load_model_to_test = True

    load_model_to_train = False #TODO: True, continue training | False, train with init_weights
    next_epoch = 19 #TODO: set as (latest checkpoint number + 1) if load_model_to_train = True

    load_model_to_test = True # ignore for now
    test_sample_size = 15

    device = "cuda" if torch.cuda.is_available() else "cpu"
    point_batches = 500
    seed = 1234

class HyperParameters:
    epochs = 20
    batch_size_train = 32
    batch_size_val = 16
    learning_rate = 0.001
    # num_workers = 1
    learning_rate_decay = 0.5

config = Configuration()
hparams = HyperParameters()
print('Device:',config.device)
torch.manual_seed(config.seed)


class CustomDataset(Dataset):
    def __init__(self, root_dir, process_type):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir)]
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
            # l_encoder_img, a_encoder_img, b_encoder_img = self.lab_encoder_img[:,:,0],self.lab_encoder_img[:,:,1],self.lab_encoder_img[:,:,2]
            l_encoder_img = self.lab_encoder_img[:,:,0]
            
            #*** Normalizing l-channel between [-1,1] ***
            l_encoder_img = l_encoder_img/50.0 - 1.0
            
            #*** Repeat the l-channel to 3 dimensions ***
            l_encoder_img = torchvision.transforms.ToTensor()(l_encoder_img)
            l_encoder_img = l_encoder_img.expand(3,-1,-1)
            
            ''' Inception Images '''
            #*** Convert the inception color image to lab space ***
            self.lab_inception_img = cv2.cvtColor(rgb_inception_img,cv2.COLOR_BGR2Lab)
            
            #*** Extract the l-channel of inception lab image *** 
            l_inception_img = self.lab_inception_img[:,:,0]/50.0 - 1.0
             
            #*** Convert the inception l-image to torch Tensor and stack it in 3 channels ***
            l_inception_img = torchvision.transforms.ToTensor()(l_inception_img)
            l_inception_img = l_inception_img.expand(3,-1,-1)
            
            ''' RGB Images '''
            rgb_encoder_img = cv2.cvtColor(rgb_encoder_img,cv2.COLOR_BGR2RGB)
            rgb_encoder_img = torchvision.transforms.ToTensor()(rgb_encoder_img)
#             rgb_encoder_img.requires_grad_(True) 
            
            return l_encoder_img, rgb_encoder_img, l_inception_img, self.files[index]
        
        except Exception as e:
            print('Exception at ',self.files[index], e)
            return torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), 'Error'
            

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
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
    
class FusionLayer(nn.Module):
    def __init__(self):
        super(FusionLayer,self).__init__()

    def forward(self, inputs, mask=None):
        ip, emb = inputs
        emb = torch.stack([torch.stack([emb],dim=2)],dim=3)
        emb = emb.repeat(1,1,ip.shape[2],ip.shape[3])
        fusion = torch.cat((ip,emb),1)
        return fusion
    
class Decoder(nn.Module):
    def __init__(self, input_depth):
        super(Decoder,self).__init__()
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
    
class Colorization(nn.Module):
    def __init__(self, depth_after_fusion):
        super(Colorization,self).__init__()
        self.encoder = Encoder()
        self.fusion = FusionLayer()
        self.after_fusion = nn.Conv2d(in_channels=1256, out_channels=depth_after_fusion,kernel_size=1, stride=1,padding=0)
        self.bnorm = nn.BatchNorm2d(256)
        self.decoder = Decoder(depth_after_fusion)

    def forward(self, img_l, img_emb):
        img_enc = self.encoder(img_l)
        # new_img_emb = torch.zeros_like(img_emb)
        fusion = self.fusion([img_enc, img_emb])
        fusion = self.after_fusion(fusion)
        fusion = self.bnorm(fusion)
        return self.decoder(fusion)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)

model = Colorization(256).to(config.device) 
optimizer = torch.optim.Adam(model.parameters(),lr=hparams.learning_rate, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)

if config.load_model_to_train or config.load_model_to_test:
    checkpoint = torch.load(f"Models/{config.model_file_name}",map_location=torch.device(config.device))
   
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device) 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k,v in state.items():
            if isinstance(v,torch.Tensor):
                state[k] = v.cuda()
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print('Loaded pretrain model | Previous train loss:',checkpoint['train_loss'])
else:
    model.apply(init_weights)

inception_model = models.inception_v3(pretrained=True).float().to(config.device)
inception_model = inception_model.float()
inception_model.eval()
# loss_criterion = torch.nn.MSELoss(reduction='mean').to(config.device)
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='mean', normalize=True).to(config.device)


""" Inference on a Sample """
# 40670 test images

def process_RGB_output(output_rgb):
    rgb_channels = output_rgb[0].cpu().detach().numpy().transpose(1,2,0) # (224,224,3)
    im_rgb_processed = (rgb_channels * 255).astype(np.uint8)
    # im_rgb_processed = cv2.cvtColor(im_rgb_processed,cv2.COLOR_BGR2RGB)
    im_rgb_processed = torchvision.transforms.ToTensor()(im_rgb_processed) # (3,224,224) # auto normalise under the hood, dont need to * 255 again
    return im_rgb_processed

# 1 batch = 1 image
def infer_sample(sample_size):
    count = 0
    data = {'file_name': [], 'img_l_encoder': [], 'output_rgb_img': [], 'loss': []}
    files = []
    for img_l_encoder, img_rgb_encoder, img_l_inception, file_name in test_dataloader:
        if count >= sample_size:
            break
        print("Processing", file_name[0])
        files.append(file_name[0])

        #*** Move data to GPU if available ***
        img_l_encoder = img_l_encoder.to(config.device)
        img_rgb_encoder = img_rgb_encoder.to(config.device)
        img_l_inception = img_l_inception.to(config.device)

        #*** Intialize Model to Eval Mode ***
        model.eval()

        #*** Forward Propagation ***
        img_embs = inception_model(img_l_inception.float())
        output_rgb = model(img_l_encoder,img_embs) # now it's actually RGB

        #*** Process model output ***
        output_rgb_processed = process_RGB_output(output_rgb)
        save_image(output_rgb_processed,'./Outputs/'+file_name[0])
        output_rgb_img = output_rgb_processed.cpu().detach().numpy()

        #*** Loss Calculation ***
        loss = lpips(output_rgb, img_rgb_encoder.float()) # pred, target

        count += 1
        
        data['file_name'].append(file_name)
        data['img_l_encoder'].append(img_l_encoder)
        data['output_rgb_img'].append(output_rgb_img)
        data['loss'].append(loss.item())
               
    return data


def display_save(file_name, img_l_encoder, output, truth, loss):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{file_name}\nloss: {loss:.8f}')
    fig.tight_layout()
    fig.subplots_adjust(top=0.83)

    # subplots
    axs[0].imshow(img_l_encoder[0][0].cpu().numpy(), cmap='gray')
    axs[0].set_title("L")
    axs[0].axis('off')
    
    axs[1].imshow(output.transpose(1,2,0))
    axs[1].set_title("output_RGB_LPIPS")
    axs[1].axis('off')
    
    axs[2].imshow(cv2.cvtColor(truth, cv2.COLOR_BGR2RGB)) 
    axs[2].set_title("truth")
    axs[2].axis('off')

    plt.show()
    # fig.savefig(f'./Compare/{file_name}', dpi=600)

test_dataset = CustomDataset(f'{config.data_path}/test2017','test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
print('Test: ',len(test_dataloader), '| Total Image:',len(test_dataloader))

sample_size = config.test_sample_size

file_name, img_l_encoder, output_rgb_img, loss = list(infer_sample(sample_size).values())


for i in range(sample_size):    
    truth_img = cv2.imread(f'{config.data_path}/test2017/{file_name[i][0]}').astype(np.float32) 
    truth_img /= 255.0 
    truth_img = cv2.resize(truth_img, (224, 224))

    # print(output_rgb_img[i])
    
    display_save(file_name[i][0], img_l_encoder[i], output_rgb_img[i], truth_img, loss[i])