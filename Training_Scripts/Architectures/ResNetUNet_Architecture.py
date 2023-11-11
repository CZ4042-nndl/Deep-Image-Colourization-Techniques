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

def find_max_epoch(root_dir, checkpoint_file_name):
    file_list = list(os.listdir(root_dir))
    max_epoch_no = 1
    max_file_name = ""
    for file_name in file_list:
        if len(file_name) > len(checkpoint_file_name) and file_name[:len(checkpoint_file_name)] == checkpoint_file_name:
            epoch_number = int(file_name[len(checkpoint_file_name): -3])
            if epoch_number > max_epoch_no:
                max_epoch_no = epoch_number
                max_file_name = file_name
    
    return max_epoch_no, max_file_name

class Configuration:
    checkpoint_file_name = 'ResNetUNet_checkpoint'  # The number will follow and then the .pt
    max_epoch_no, max_file_name = find_max_epoch("Models/", checkpoint_file_name)
    load_model_file_name = max_file_name
    starting_epoch = 1  # + max_epoch_no
    load_model_to_train = False
    load_model_to_test = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    point_batches = 500

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

if not config.load_model_to_test:
    train_dataset = CustomDataset('coco2017/train2017','train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size_train, shuffle=True)

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


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)

model = Colorization_ResNetUNet(2).to(config.device) 
optimizer = torch.optim.Adam(model.parameters(),lr=hparams.learning_rate, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)

if config.load_model_to_train or config.load_model_to_test:
    checkpoint = torch.load("Models/"+config.load_model_file_name, map_location=torch.device(config.device))
   
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

# inception_model = models.inception_v3(pretrained=True).float().to(config.device)
# inception_model = inception_model.float()
# inception_model.eval()
loss_criterion = torch.nn.MSELoss(reduction='mean').to(config.device)

if not config.load_model_to_test:
    train_dataset = CustomDataset('coco2017/train2017','train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size_train, shuffle=True)
    

    validataion_dataset = CustomDataset('coco2017/val2017','validation')
    validation_dataloader = torch.utils.data.DataLoader(validataion_dataset, batch_size=hparams.batch_size_val, shuffle=False)

if not config.load_model_to_test:
    print('Train:',len(train_dataloader), '| Total Images:',len(train_dataloader)*hparams.batch_size_train)
    # print('Valid:',len(validation_dataloader), '| Total Images:',len(validation_dataloader)*hparams.batch_size_val)

if not config.load_model_to_test:
    for epoch in tqdm(range(config.starting_epoch,hparams.epochs)):
        print('Starting epoch:', epoch)

        #*** Training step ***
        loop_start = time.time()
        avg_loss = 0.0
        batch_loss = 0.0
        main_start = time.time()
        model.train()

        for idx,(img_l_encoder, img_ab_encoder, img_l_inception, img_rgb, file_name) in enumerate(train_dataloader):
            #*** Skip bad data ***
            if not img_l_encoder.ndim:
                continue

            #*** Move data to GPU if available ***
            img_l_encoder = img_l_encoder.to(config.device)
            img_ab_encoder = img_ab_encoder.to(config.device)
            img_l_inception = img_l_inception.to(config.device)

            #*** Initialize Optimizer ***
            optimizer.zero_grad()

            #*** Forward Propagation ***
            # img_embs = inception_model(img_l_inception.float())
            output_ab = model(img_l_encoder)

            #*** Back propogation ***
            loss = loss_criterion(output_ab, img_ab_encoder.float())
            loss.backward()

            #*** Weight Update ****
            optimizer.step()

            #*** Loss Calculation ***
            avg_loss += loss.item()
            batch_loss += loss.item()

            #*** Print stats after every point_batches ***
            if idx%config.point_batches==0: 
                loop_end = time.time()   
                print('Batch:',idx, '| Processing time for',config.point_batches,':',loop_end-loop_start,'s | Batch Loss:', (batch_loss/config.point_batches)*100)
                loop_start = time.time()
                batch_loss = 0.0

        #*** Print Training Data Stats ***
        train_loss = avg_loss/len(train_dataloader)*hparams.batch_size_train
        print('Training Loss:',train_loss,'| Processed in ',time.time()-main_start,'s')

        scheduler.step(train_loss)

        #*** Save the Model to disk ***
        model_file_name = "Models/" + config.checkpoint_file_name + str(epoch) + ".pt"
        checkpoint = {'model_state_dict': model.state_dict(),\
                      'optimizer_state_dict' : optimizer.state_dict(),
                      'scheduler_state_dict' : scheduler.state_dict(),
                      'train_loss':train_loss}
        torch.save(checkpoint, model_file_name)
        print("Model saved at:", model_file_name)
        # torch.cuda.empty_cache()

test_dataset = CustomDataset('coco2017/test2017','test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
print('Test: ',len(test_dataloader), '| Total Image:',len(test_dataloader))

def concatente_and_colorize(im_lab, img_ab):
    # Assumption is that im_lab is of size [1,3,224,224]
    #print(im_lab.size(),img_ab.size())
    np_img = im_lab[0].cpu().detach().numpy().transpose(1,2,0)
    lab = np.empty([*np_img.shape[0:2], 3],dtype=np.float32)
    lab[:, :, 0] = np.squeeze(((np_img + 1) * 50))
    lab[:, :, 1:] = img_ab[0].cpu().detach().numpy().transpose(1,2,0) * 127
    np_img = cv2.cvtColor(lab,cv2.COLOR_Lab2RGB) 
    color_im = torch.stack([torchvision.transforms.ToTensor()(np_img)],dim=0)
    return color_im

#*** Inference Step ***
avg_loss = 0.0
loop_start = time.time()
batch_start = time.time()
batch_loss = 0.0

for idx,(img_l_encoder, img_ab_encoder, img_l_inception, img_rgb, file_name) in enumerate(test_dataloader):
        #*** Skip bad data ***
        if not img_l_encoder.ndim:
            continue
            
        #*** Move data to GPU if available ***
        img_l_encoder = img_l_encoder.to(config.device)
        img_ab_encoder = img_ab_encoder.to(config.device)
        img_l_inception = img_l_inception.to(config.device)
        
        #*** Intialize Model to Eval Mode ***
        model.eval()
        
        #*** Forward Propagation ***
        # img_embs = inception_model(img_l_inception.float())
        output_ab = model(img_l_encoder)
        
        #*** Adding l channel to ab channels ***
        color_img = concatente_and_colorize(torch.stack([img_l_encoder[:,0,:,:]],dim=1),output_ab)
        color_img_jpg = color_img[0].detach().numpy().transpose(1,2,0)
        # cv2.imshow(color_img_jpg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite('outputs/'+file_name[0],color_img_jpg*255)
        save_image(color_img[0],'Outputs/'+file_name[0])

#       #*** Printing to Tensor Board ***
        grid = torchvision.utils.make_grid(color_img)
        
        #*** Loss Calculation ***
        loss = loss_criterion(output_ab, img_ab_encoder.float())
        avg_loss += loss.item()
        batch_loss += loss.item()

        if idx%config.point_batches==0: 
            batch_end = time.time()   
            print('Batch:',idx, '| Processing time for',config.point_batches,':',str(batch_end-batch_start)+'s', '| Batch Loss:', batch_loss/config.point_batches)
            batch_start = time.time()
            batch_loss = 0.0
        
test_loss = avg_loss/len(test_dataloader)
print('Test Loss:',avg_loss/len(test_dataloader),'| Processed in ',str(time.time()-loop_start)+'s')

