import torch
import clip
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import nn, optim
from torchvision import transforms, utils
import os
import numpy as np
from copy import deepcopy
from torch.optim import lr_scheduler

clsname=[
'76', 'Act II', 'Admiral', 'alpinestars', 'ASUS', 'ATTENIR', 'avery dennison', 'Bacardi', 'battleship', "Bellamy's Australia", 'benuron', 'Bob Evans Restaurants', 'Bold Rock Hard Cider', 'Bovril', 'Brothers Cider', 'Bubba Gump Shrimp Company', 'Bubblicious', 'Bulls-Eye Barbecue', 'Burger King', 'Cafe Coffee Day', 'chatime', 'chewits', 'CoolPAD', 'De Cecco', 'Dongeejiao', 'Enfamil', "Fox's Biscuits", 'g.i. joe', 'Haizhilan', 'Haribo', 'Heineken', 'Jordans', 'Jus-Rol', 'kleenex', 'lexus', 'maybach', 'mclaren', 'new balance', 'new holland', 'playmobil', 'prismacolor', 'prudenial', 'quickchek', 'regina', 'Schiff', 'sherrin', 'shiseido', 'skype', 'staedtler', 'steeden', 'thomapyrin', 'Tim Tam', 'villa zamorano', 'violet crumble', 'vision street wear', 'Vitafusion', 'Wanzaimatou', 'wild berry skittles', 'yorkshire tea', 'zara', 'zendium', 'Taitaile', 'Aveda', 'Aveeno', 'Dr. Oetker', 'La Vie', 'Hancock', 'Impact', 'Sessions', 'Hormel', 'Trust', 'Ace', 'G.A.S', 'IFA', 'The California Raisins', 'JD', 'SPC', 'Crown', 'MiO', 'Heat', 'Angelina', 'Sizzler', 'Orion', 'Pixian', 'Speakeasy', 'Tesla', 'Brunswick', "Mack & Dave's Department Store", 'Westminster', 'Alcon', 'Bona', 'BreadTalk', 'JMP', "Nelsons's", 'RCA', 'Red Barn', 'Sea & Sea', 'KR3W', 'Ola', 'Prodent'
]


class Image_caption_dataset(Dataset):
    def __init__(self, df, preprocess):
        self.images = df["image"]
        self.caption = df["caption"]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        images = self.preprocess(Image.open(self.images[idx]).convert('RGB'))
        caption = self.caption[idx]
        return images, caption
        

def load_data(batch_size, preprocess):
    
    dataroot = '/home/sdnu2022/logo100_32/images'    
    txt_path = '/home/sdnu2022/桌面/zjx_code/Logo_FSCIL/Logo100_32/baselines/SAVC_CROSS/data/index_list/mini_imagenet/session_1.txt'

    index=[]
    lines = [x.strip() for x in open(txt_path, 'r').readlines()]
    for line in lines:
        index.append(line.split('/')[1]+'/'+line.split('/')[2])
    df = {'image': [], 'caption':[]}
    
    a=0
    for i in index:
        img_path = os.path.join(dataroot, i)
        df['image'].append(img_path)
        k = int(a)/100 					#数字根据每个类样本数量改变
        caption = "a logo photo of a "+str(clsname[int(k)])	
        df['caption'].append(caption)
        a=a+1

    dataset = Image_caption_dataset(df, preprocess)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True, pin_memory=True)
    return train_dataloader


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()
        

def evaluate_model(model, dataloader, loss_img, loss_txt, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            list_image, list_txt = batch
            texts = clip.tokenize(list_txt).to(device)
            images = list_image.to(device)
            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(len(list_image), dtype=torch.long, device=device)

            # Calculate accuracy
            total_samples += len(ground_truth)
            total_correct += ((torch.argmax(logits_per_image, dim=1) == ground_truth) & 
                              (torch.argmax(logits_per_text, dim=1) == ground_truth)).sum().item()

    accuracy = total_correct / total_samples
    return accuracy


         
# parameter
epoch = 30
batch_size = 32
 
# load clip model
device = "cuda:1" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)  # Must set jit=False for training
 
train_dataloader = load_data(batch_size, preprocess)

clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16
 
optimizer = optim.Adam(model.parameters(), lr=5e-6, betas=(0.9, 0.98), eps=1e-6 ,weight_decay=0.001) #5e-6已知起点50多随后会掉点？
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) 

# 设置参数
loss_img = nn.CrossEntropyLoss().to(device)
loss_txt = nn.CrossEntropyLoss().to(device)

best_accuracy = 0.0
best_model_state = None
phase = "train"


for i in range(epoch):
    scheduler.step()
    for j, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == "train"):
            list_image, list_txt = batch
            texts = clip.tokenize(list_txt).to(device)
            images = list_image.to(device)
            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(len(list_image), dtype=torch.long, device=device)

            # 反向传播
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            #print(f"epoch [{i}]-[{j}]: {total_loss.item()}")
            print(f"epoch [{i}]-[{j}]")
        
    accuracy = evaluate_model(model, train_dataloader, loss_img, loss_txt, device)
    print(f"Epoch [{i}]: Accuracy: {accuracy}")

    # Save the best model state
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state = deepcopy(model.state_dict())

model.load_state_dict(best_model_state)

torch.save(model, '/home/sdnu2022/桌面/zjx_code/Logo_FSCIL/Logo100_32/baselines/SAVC_CROSS/model_minilogo_vitb16.pkl')
print(f"model have saved")


