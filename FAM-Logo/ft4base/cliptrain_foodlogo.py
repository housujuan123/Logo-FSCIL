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
'Perdue', 'Asahi', 'Appleton Estate', 'Horlicks', 'Heinz Baked Beans', 'Baja Fresh', "Lay's", 'waffle crisp', 'Burger King', 'yili', 'Harvest Crunch', 'mccafe', 'evian', 'peptamen', "Newman's Own", "Gregg's", 'second cup', 'jack in the box', 'Chocapic', 'winiary', 'Maestro Dobel', 'Chocolate Lucky Charms', 'molson brador', 'Brothers Cider', 'Boca Burger', 'sweetened wheat-fuls', 'Highland Spring', 'Hard Rock Cafe', 'Cascadian Farm', 'Maruchan', 'Cibo Espresso', 'Cafe du Monde', 'Michel et Augustin', 'Beer Nuts', 'Cocoa Puffs', 'Mecca Cola', "Bush's Best", 'Carling Black Label', 'vladivar', 'Straus', 'Four Seas Ice Cream', "McDonald's", 'stadium mustard', 'Mondelez International', 'Baxters', 'xifeng', 'Bahlsen', 'robust', 'Celestial Seasonings', 'Britannia', 'chicago town', 'Hortex', 'BreadTalk', 'nesvita', 'CHIVAS', 'Alter Eco', 'Darden-1', "Brigham's Ice Cream", "Maker's Mark", 'Hormel', "Eegee's", 'Hollys Coffee', 'copella', 'strawberry rice krispies', 'Frosty Boy-1', 'lan-choo', 'Gobstoppers', 'Burgerville', 'Bellywashers', 'Mengzhilan', "Mike's", 'Bon Pari', 'FruChocs', 'Cocoa Pebbles', 'Bicerin', 'Tyrrells', "tully's coffee", 'Aqua Carpatica', "Fox's Pizza Den", 'Honey Bunches of Oats', 'Divine', 'Cracker', 'sveltesse', "Kwality Wall's", 'laura secord chocolates', 'Efes', 'Canada Dry', 'ready brek', 'Buondi', 'nutren', 'La Choy', 'Fruco', 'Heinz Tomato Ketchup', 'Ketel One', "romano's macaroni grill", 'ricoffy', 'Ciego Montero', 'Irn Bru Bar', 'Cestbon', 'Cafe HAG'
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
    
    dataroot = '/home/sdnu2022/foodlogo_200/images'    
    txt_path = '/home/sdnu2022/桌面/zjx_code/Logo_FSCIL/foodlogo200/baselines/SAVC_CROSS/data/index_list/mini_imagenet/session_1.txt'

    index=[]
    lines = [x.strip() for x in open(txt_path, 'r').readlines()]
    for line in lines:
        index.append(line.split('/')[1]+'/'+line.split('/')[2])
    df = {'image': [], 'caption':[]}
    
    a=0
    for i in index:
        img_path = os.path.join(dataroot, i)
        df['image'].append(img_path)
        k = int(a)/30 					#数字根据每个类样本数量改变
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
device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)  # Must set jit=False for training
 
train_dataloader = load_data(batch_size, preprocess)

clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

# Params used from paper, the lr is smaller, more safe for fine tuning to new dataset 
optimizer = optim.Adam(model.parameters(), lr=5e-6, betas=(0.9, 0.98), eps=1e-6 ,weight_decay=0.001) #5e-6=上来就85的精度，之后就再也上不去掉点
scheduler = lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1) 

# 设置参数
loss_img = nn.CrossEntropyLoss().to(device)
loss_txt = nn.CrossEntropyLoss().to(device)

best_accuracy = 0.0
best_model_state = None
phase = "train"

accuracy = evaluate_model(model, train_dataloader, loss_img, loss_txt, device)
print(f"before start : Accuracy: {accuracy}")

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
torch.save(model, '/home/sdnu2022/桌面/zjx_code/Logo_FSCIL/foodlogo200/baselines/SAVC_CROSS/model_minilogo_vitb16.pkl')
print(f"model have saved")


