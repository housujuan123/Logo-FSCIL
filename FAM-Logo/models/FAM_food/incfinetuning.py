import os.path as osp
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader

from .helper import *
from utils import *
from dataloader.data_utils import *
from dataloader.sampler import Image_caption_dataset, LinearWarmupScheduler

#----------
import clip
from torch import optim
import pandas as pd
from PIL import Image
import os

clsname=[
'Perdue', 'Asahi', 'Appleton Estate', 'Horlicks', 'Heinz Baked Beans', 'Baja Fresh', "Lay's", 'waffle crisp', 'Burger King', 'yili', 'Harvest Crunch', 'mccafe', 'evian', 'peptamen', "Newman's Own", "Gregg's", 'second cup', 'jack in the box', 'Chocapic', 'winiary', 'Maestro Dobel', 'Chocolate Lucky Charms', 'molson brador', 'Brothers Cider', 'Boca Burger', 'sweetened wheat-fuls', 'Highland Spring', 'Hard Rock Cafe', 'Cascadian Farm', 'Maruchan', 'Cibo Espresso', 'Cafe du Monde', 'Michel et Augustin', 'Beer Nuts', 'Cocoa Puffs', 'Mecca Cola', "Bush's Best", 'Carling Black Label', 'vladivar', 'Straus', 'Four Seas Ice Cream', "McDonald's", 'stadium mustard', 'Mondelez International', 'Baxters', 'xifeng', 'Bahlsen', 'robust', 'Celestial Seasonings', 'Britannia', 'chicago town', 'Hortex', 'BreadTalk', 'nesvita', 'CHIVAS', 'Alter Eco', 'Darden', "Brigham's Ice Cream", "Maker's Mark", 'Hormel', "Eegee's", 'Hollys Coffee', 'copella', 'strawberry rice krispies', 'Frosty Boy', 'lan-choo', 'Gobstoppers', 'Burgerville', 'Bellywashers', 'Mengzhilan', "Mike's", 'Bon Pari', 'FruChocs', 'Cocoa Pebbles', 'Bicerin', 'Tyrrells', "tully's coffee", 'Aqua Carpatica', "Fox's Pizza Den", 'Honey Bunches of Oats', 'Divine', 'Cracker', 'sveltesse', "Kwality Wall's", 'laura secord chocolates', 'Efes', 'Canada Dry', 'ready brek', 'Buondi', 'nutren', 'La Choy', 'Fruco', 'Heinz Tomato Ketchup', 'Ketel One', "romano's macaroni grill", 'ricoffy', 'Ciego Montero', 'Irn Bru Bar', 'Cestbon', 'Cafe HAG', 'Meiji', 'Kotipizza', 'Cherry 7Up', 'Badia', 'Johnny Rockets', 'Huy Fong Foods', 'Balaji Wafers', 'Apple Zings', "Bewley's", 'Carling', 'Bigg Mixx', 'Coco Pops', 'APEROL', 'Jelly Belly', 'Calbee', 'Cielo', 'Bridgehead Coffee', 'lion cereal', 'Chipsmore', 'Grapette', 'toblerone', 'Bartolo Nardini', 'smirnoff', 'Burger Street', 'wuyutai', "rosati's", 'Cha Dao', "Fox's Biscuits", 'chex mix', 'Poulain', 'Laciate', 'Danone', 'Dalda', 'ledo pizza', 'Maggi Masala noodles', 'CDO Foodsphere', 'Highland Toffee', 'Goya', 'Idaho SPUD', 'screaming yellow zonkers', 'maggi noodles', 'Gay Lea', 'Cocosette', 'Bisquick', "ching's secret", 'Frosty Jack Cider', 'NISSIN', 'Monsanto', 'Chocomel', "Haldiram's", "Graeter's", 'nestle', 'tooty footies', 'oberweis dairy', "Hershey's Cookies 'n' Creme", "timothy's world coffee", 'williams fresh cafe', 'Coco Roos', 'COFCO', 'DEVONDALE', 'Gordon Food Service', 'Cape Cod Potato Chips', "Bruegger's", 'nestle corn flakes', 'Carapelli', 'Greene King', "Lady's Choice", 'Freihofers', 'la saltena', 'Frosted Shredded Wheat', 'Brugal', 'Chips Ahoy', 'Iams', 'Fujiya', 'Buca di Beppo', 'Qianhe', 'Creemore', 'obaras', 'st arnou', 'Hot Pockets', 'Laffy Taffy', 'pisco porton', 'Chronic Tacos', 'Haribo', 'Frosted Mini Spooners', 'Juicy Fruit', 'Beaulieu', 'Maltesers', 'Polar', 'Eden Cheese', 'Boon Rawd Brewery', 'Magners lrish', 'Jollibee', 'Chef Boyardee', 'robeks', "Kern's", 'Coors', "steve's ice cream", 'Capri Sun', 'Bega']

incmapping = {
        100: 0, 101: 1, 102: 2, 103: 3, 104: 4, 105: 5, 106: 6, 107: 7, 108: 8, 109: 9,
        110: 0, 111: 1, 112: 2, 113: 3, 114: 4, 115: 5, 116: 6, 117: 7, 118: 8, 119: 9,
        120: 0, 121: 1, 122: 2, 123: 3, 124: 4, 125: 5, 126: 6, 127: 7, 128: 8, 129: 9,
        130: 0, 131: 1, 132: 2, 133: 3, 134: 4, 135: 5, 136: 6, 137: 7, 138: 8, 139: 9,
        140: 0, 141: 1, 142: 2, 143: 3, 144: 4, 145: 5, 146: 6, 147: 7, 148: 8, 149: 9,
        150: 0, 151: 1, 152: 2, 153: 3, 154: 4, 155: 5, 156: 6, 157: 7, 158: 8, 159: 9,
        160: 0, 161: 1, 162: 2, 163: 3, 164: 4, 165: 5, 166: 6, 167: 7, 168: 8, 169: 9,
        170: 0, 171: 1, 172: 2, 173: 3, 174: 4, 175: 5, 176: 6, 177: 7, 178: 8, 179: 9,
        180: 0, 181: 1, 182: 2, 183: 3, 184: 4, 185: 5, 186: 6, 187: 7, 188: 8, 189: 9,
        190: 0, 191: 1, 192: 2, 193: 3, 194: 4, 195: 5, 196: 6, 197: 7, 198: 8, 199: 9,
}



def incfinetune(cmodel, cprocessor, trainloader, session, args, num_classes, best_former_dict=None):
    epoch = 10    
    batch_size = 50 
    m=5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lastincclsnum = num_classes-10    
    
    trainlist_vis = [] 
    trainlist_vis = [[] for _ in range(10)]  
    for i, batch in enumerate(trainloader):
        _, labels, image_path = batch[0], batch[1].cuda(), batch[2]
        for j in range(len(image_path)):
            image = Image.open(image_path[j])
            image_input = cprocessor(image).unsqueeze(0).to(device)                
            image_feature = cmodel.encode_image(image_input)
            class_label = labels[j].item()  
            index = int(j/5)
            trainitem = {"image": image_feature.to(device), "label": class_label}   
            trainlist_vis[index].append(trainitem)

    trainlist_text = []
    trainloader_text = load_data(batch_size, session, args)  
    text_features = torch.empty((0, 512), dtype=torch.float32)
    text_labels = None
    for i, batch in enumerate(trainloader_text):
        tdata, tlabel = batch[0], batch[1]
        text_input = process_text_tuple(tdata).to(device)  
        text_feature = cmodel.encode_text(text_input).to(device)
        if text_features.size(0) == 0:   
            text_features = text_feature.float() 
        else:
            text_features = torch.cat((text_features, text_feature.float()), dim=0)
        
        for j in range(text_feature.shape[0]):
            featuree = text_feature[j]
            labell = tlabel[j]
            trainitem = {"image": featuree.unsqueeze(0), "label": labell}    
            trainlist_text.append(trainitem)
        text_labels = tlabel   

    linear_head = nn.Linear(512, 10, bias=False) 
    linear_head.weight.data = get_zero_shot_weights(trainloader_text, 10, 512, cmodel, device="cuda") 
    logit_head = LogitHead(linear_head,logit_scale=4.60517).train().to(device)

    params_groups = [{'params': logit_head.parameters()}]
  	  
    optimizer = torch.optim.AdamW(params_groups, lr=0.0001, weight_decay=0.01, betas=(0.9, 0.999))   
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(12800))  
    scheduler = LinearWarmupScheduler(optimizer, scheduler, 50, 1e-5)
    criterion = torch.nn.CrossEntropyLoss()  
    
    best_acc = float('inf')
    best_proto_dict = None
  
    correct_count = 0
    total_count = 0
    
    for i in tqdm(range(epoch)):
        logit_head.train()
  
        all_features = text_features
        all_labels = text_labels.type(torch.long)
  	    
        for j in range(len(trainlist_vis)):
            for k in range(len(trainlist_vis[j])):   
                image_feature = trainlist_vis[j][k]["image"]
                image_label = trainlist_vis[j][k]["label"]
                image_label = torch.tensor([image_label])
                all_features = torch.cat([all_features, image_feature], dim=0)
                all_labels= torch.cat([all_labels, image_label], dim=0)

        logit = logit_head(all_features).to(device) 
        
        logitss = torch.argmax(logit, dim=1)
        mapped_logit = logitss.clone()
        mapped_logit = mapped_logit.unsqueeze(0)
        mapped_logit = mapped_logit.to(device)                         
        mapped_labels = all_labels.clone()
        for key, value in incmapping.items():
            mapped_labels[all_labels == key] = value
        mapped_labels = mapped_labels.to(device)

        correct_count += torch.sum(mapped_logit == mapped_labels).item()
        total_count += all_labels.shape[0]  
        train_acc = correct_count / total_count
        #print("train_acc:",train_acc)
  	    
        if train_acc > best_acc:
            best_acc = train_acc
            best_head_dict = deepcopy(logit_head.state_dict())
            logit_head.load_state_dict(best_head_dict)
        
        loss = criterion(logit, mapped_labels)  #torch.nn.CrossEntropyLoss()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
    
    all_head = logit_head	 
    if session>1: 
        all_head = nn.Linear(512, num_classes, bias=False) 
        all_head = LogitHead(all_head,logit_scale=4.60517).to(device)  
        all_head.head.weight.data[:lastincclsnum] = best_former_dict.head.weight.data
        all_head.head.weight.data[lastincclsnum:] = logit_head.head.weight.data

    return all_head
    
#new    
def load_data(batch_size, session, args):

    txt_path = "data/index_list/mini_imagenet/session_" + str(session + 1) + '.txt' 
    index=[]
    lines = [x.strip() for x in open(txt_path, 'r').readlines()]
    for line in lines:
        index.append(line.split('/')[2])
        
    df = {'text': [], 'label':[]}
    a=0
    for i in index:
        if a%5 == 0:
            k = int(args.base_class)+(int(session)-1)*10+(int(a)/5)   
            caption = "a logo photo of a "+str(clsname[int(k)])   
            df['text'].append(caption)
            df['label'].append(int(k))
        a=a+1
    dataset = Image_caption_dataset(df) 
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    
    return train_dataloader
    
    
def process_text_tuple(text_tuple):
    tokenized_texts = [clip.tokenize(text) for text in text_tuple]
    concatenated_texts = torch.cat(tokenized_texts)
    
    return concatenated_texts

    
def get_zero_shot_weights(trainloader_t, num_classes, in_features, cmodel, device="cuda"):
    with torch.no_grad():
        weights = torch.zeros(num_classes, in_features)
        textftemp = torch.zeros(num_classes, in_features)
        classindex = []
        for i, batch in enumerate(trainloader_t): 
            tdata, tlabel = batch[0], batch[1]
            #print('data:',tdata,'label:',tlabel)
            texts = None
            for j in range(len(tdata)):
                texts = clip.tokenize(tdata[j]).to(device)
                text_features = cmodel.encode_text(texts).float()
                weights[j] = text_features.to(device)
           
        weights.data = torch.nn.functional.normalize(weights, dim=1)
        return weights
    
class LogitHead(nn.Module):
    def __init__(self, head, logit_scale=float(np.log(1 / 0.07))):
        super().__init__()
        self.head = head
        self.logit_scale = logit_scale
        
        # Not learnable for simplicity
        self.logit_scale = torch.FloatTensor([logit_scale]).cuda()

    def forward(self, x):
        x = F.normalize(x, dim=1)
        x = self.head(x)
        x = x * self.logit_scale.exp()
        return x
