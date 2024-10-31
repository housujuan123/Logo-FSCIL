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
'24seven', 'A1 Steak Sauce', 'ABLE', 'Abuelita', 'aceh', 'Addidas', 'Aero Contractors', 'Algida', 'Alpen Gold', 'Amkette', 'Ampeg', "Amy's Ice Creams", 'Apple Jacks', 'Aprilia', 'Aquafresh', 'ASCASO', 'Ashbury', 'Astro', 'Asusu', 'Atlantic Airlines', 'Backwoods', 'BAE', 'Bahia', 'Baker Skateboards', 'Bakers', 'Banana Frosted Flakes', 'Baraka', 'Batman', 'Billabong', 'Blackthorn Cider', 'Blaser', 'Blue Bonnet', 'Blue Stratos', "Boscov's", 'Brazil', 'Bridgestone', 'Bristol', 'Brooklyn', 'Buffalo', 'Buffalo Rock Ginger Ale', "Caffrey's", 'Cailler', 'Cain', 'Caloi', 'Cape Cod Potato Chips', 'Caress', 'Carrefour', 'Carte Noire', 'Centurion', 'cestbon', 'CHEF', 'Chery', 'Chiclets', "Chili's", 'CHINA CITIC', 'Chocapic', 'CHS', "Chuck E. Cheese's", 'Cigar City', 'Circa', 'Clinic', 'Coco Pops', 'Coffee-Mate', 'Connaught', 'Connector', 'Contour', 'Cool Whip', 'Diet Coke', 'Dimension', 'Dollar', "Domino's Pizza", 'Double Take', "Dr Brown's", 'DUX', 'DVS Shoes', 'Eclipse', 'El Jimador', 'Ellesse', 'eMachines', 'Escada', 'Eureka', 'Expert', 'Fram', 'Frito-Lay', 'Fujitsu', 'Full Moon', 'Geely', 'Gel', 'Gillette', 'Glock', 'GoPro', 'Granada', 'Granville', 'Gull', 'Guojiao', "Hartley's", 'Heartland', 'Highland Toffee', 'Honey Kix', 'Honey Nut Cheerios', "Howard Johnson's", 'Hummel', 'Husqvarna', 'Ibex', 'Ilford Photo', 'Intermarche', 'Irn Bru Bar', 'ITS', 'Ivory', "Jason's Deli", 'KEF', 'Kingdom', 'Kodak', 'Koenigsegg', 'Koko Krunch', 'Kopparbergs', 'Krush', 'KTM', 'Lacto', 'Lada', 'Laffy Taffy', 'Lapierre', 'Lean Cuisine', 'LEE KUM KEE', 'Lefevre-Utile', 'Lego', 'Lemonade', 'Limca', 'Liptonice', "LLoyd's", 'logo Bubblicious', 'Lotus Cars', 'Love is', 'Loyd', 'Lynx', 'MAD DOG', 'Magners Irish', 'Maple', 'Mattel', 'Maxum', 'Maxwell House', 'Mazda', 'Metz', 'Mickey Mouse', 'Mighty Dog', 'MiO', 'Mirage', 'Mirinda', 'Mitsubishi', 'Monster Munch', 'Mr Kipling', 'Mrs. Fields', 'Multi-Bran Chex', 'Murphy', 'Naleczowianka', 'NAN HA', 'Nerds', 'Nesfruta', 'Nestum', "Nice 'n Easy", 'North Coast', 'Nostromo', "O'Charley's", 'Orange Blossom', 'Orangina', 'Orbit', 'Origen', 'Orion', 'Oronamin C', 'Overture', 'Pathe', 'PDV', 'Pechoin', "Peet's Coffee & Tea", 'PEL', 'pepsi', 'Petro', 'Petronas', 'Philips', 'Pioneer', 'Pixian', 'Primos', 'ProTech', 'Proto', 'Quasar', 'Realtor', 'Red Barn', 'Red Hot & Blue', 'Red Oak', 'REFOSIAN', 'Refreshers Gums', 'Revolution', 'Reynolds', "Richard Petty 43's", 'Rip Curl', 'Russell', 'Saladworks', 'Sbarro', 'Screamers', 'Sharp', 'Solex', 'sony', 'Spider-Man', 'SR', 'Stanley', 'Star Energy', 'Steam Whistle', 'Stella', 'Sure', 'Tatiana', 'TECO', 'The Batman', 'thirst cola', 'Toby', 'Toscano', 'Total', 'Transcom', 'Trek Bicycle', 'Triaminic', 'Trio', 'Twist', 'Tylenol', 'Valero', 'Vegemite', 'Vestel', 'wanglaoji', 'Wegmans', 'Williams', 'WMF', 'Xurisheng'
]

incmapping = {
        130: 0, 131: 1, 132: 2, 133: 3, 134: 4, 135: 5, 136: 6, 137: 7, 138: 8, 139: 9,
        140: 0, 141: 1, 142: 2, 143: 3, 144: 4, 145: 5, 146: 6, 147: 7, 148: 8, 149: 9,
        150: 0, 151: 1, 152: 2, 153: 3, 154: 4, 155: 5, 156: 6, 157: 7, 158: 8, 159: 9,
        160: 0, 161: 1, 162: 2, 163: 3, 164: 4, 165: 5, 166: 6, 167: 7, 168: 8, 169: 9,
        170: 0, 171: 1, 172: 2, 173: 3, 174: 4, 175: 5, 176: 6, 177: 7, 178: 8, 179: 9,
        180: 0, 181: 1, 182: 2, 183: 3, 184: 4, 185: 5, 186: 6, 187: 7, 188: 8, 189: 9,
        190: 0, 191: 1, 192: 2, 193: 3, 194: 4, 195: 5, 196: 6, 197: 7, 198: 8, 199: 9,
        200: 0, 201: 1, 202: 2, 203: 3, 204: 4, 205: 5, 206: 6, 207: 7, 208: 8, 209: 9,
        210: 0, 211: 1, 212: 2, 213: 3, 214: 4, 215: 5, 216: 6, 217: 7, 218: 8, 219: 9,
        220: 0, 221: 1, 222: 2, 223: 3, 224: 4, 225: 5, 226: 6, 227: 7, 228: 8, 229: 9
		}


def incfinetune(cmodel, cprocessor, trainloader, session, args, num_classes, best_former_dict=None):
    # ----设置参数
    epoch = 0 #8.12对比注释      
    batch_size = 50  #一次inc session总样本量
    m=5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lastincclsnum = num_classes-10     #int(num_classes-10)  
    
    #----提取图像特征
    trainlist_vis = [] 
    trainlist_vis = [[] for _ in range(10)]  #记录图像训练数据，初始化10个空列表
    for i, batch in enumerate(trainloader):
        _, labels, image_path = batch[0], batch[1].cuda(), batch[2]
        for j in range(len(image_path)):
            image = Image.open(image_path[j])
            image_input = cprocessor(image).unsqueeze(0).to(device)       #原图提取特征                            
            image_feature = cmodel.encode_image(image_input)
            class_label = labels[j].item()  
            index = int(j/5)
            trainitem = {"image": image_feature.to(device), "label": class_label}    #图像信息上device
            trainlist_vis[index].append(trainitem)
    #print(len(trainlist_vis))

    #---获取文本特征
    trainlist_text = []
    trainloader_text = load_data(batch_size, session, args)  #文本label集
    text_features = torch.empty((0, 512), dtype=torch.float32)
    text_labels = None
    for i, batch in enumerate(trainloader_text):
        tdata, tlabel = batch[0], batch[1]
        text_input = process_text_tuple(tdata).to(device)  #tdata多数据所以用这处理方式
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
        text_labels = tlabel   #就一个patch是不是这样就行
      #到时候需要检查这个trainlist_text的形状分布
    #print(len(trainlist_text))

    #---线性层
    linear_head = nn.Linear(512, 10, bias=False) 
    linear_head.weight.data = get_zero_shot_weights(trainloader_text, 10, 512, cmodel, device="cuda") #初始化权重
    logit_head = LogitHead(linear_head,logit_scale=4.60517).train().to(device)
	

  	#---优化项 Create the optimizer----
    params_groups = [                               
        {'params': logit_head.parameters()},
  	]
  	  
    optimizer = torch.optim.AdamW(params_groups, lr=0.0001, weight_decay=0.01, betas=(0.9, 0.999))   
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(12800))  
    scheduler = LinearWarmupScheduler(optimizer, scheduler, 50, 1e-5)
    criterion = torch.nn.CrossEntropyLoss()  

    #----------------------------训练过程-------------------------------------
    
    best_acc = float('inf')
    best_proto_dict = None
  
    correct_count = 0
    total_count = 0
    
    #for i in range(iters):
    for i in tqdm(range(epoch)):
        logit_head.train()
  
        all_features = text_features
        all_labels = text_labels.type(torch.long)
        """
        print(type(all_features))
        print(all_labels)
        print(type(all_labels[0]))
        """
  	    
        for j in range(len(trainlist_vis)):
            for k in range(len(trainlist_vis[j])):   #同类的数条数据
                image_feature = trainlist_vis[j][k]["image"]
                image_label = trainlist_vis[j][k]["label"]
                #print(image_label)
                image_label = torch.tensor([image_label])
                all_features = torch.cat([all_features, image_feature], dim=0)
                all_labels= torch.cat([all_labels, image_label], dim=0)
                #print(all_labels)

        logit = logit_head(all_features).to(device)   #----分类器传入特征
        
        logitss = torch.argmax(logit, dim=1)
        mapped_logit = logitss.clone()
        # 将结果转换成形状为[1, 60]的张量
        mapped_logit = mapped_logit.unsqueeze(0)
        #print(mapped_logit)
        """
        for key, value in incmapping.items():
  			    mapped_logit[mapped_logit == key] = value
        print(mapped_logit) 
        """
        mapped_logit = mapped_logit.to(device)
                                   
        mapped_labels = all_labels.clone()
        for key, value in incmapping.items():
            mapped_labels[all_labels == key] = value
        mapped_labels = mapped_labels.to(device)
        
  
        #print(mapped_labels)
        correct_count += torch.sum(mapped_logit == mapped_labels).item()
        total_count += all_labels.shape[0]  
        train_acc = correct_count / total_count
        #print("train_acc:",train_acc)
  	    
        if train_acc > best_acc:
            best_acc = train_acc
            best_head_dict = deepcopy(logit_head.state_dict())
            # load best model
      			#image_encoder.load_state_dict(result_dict["image_encoder"])
      			#text_encoder.load_state_dict(result_dict["text_encoder"])
            logit_head.load_state_dict(best_head_dict)
        
        loss = criterion(logit, mapped_labels)  #torch.nn.CrossEntropyLoss()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
    
    all_head = logit_head	 
    if session>1:  #进行权重拼接:目前就是每个线性层和旧的线性层权重的单纯合并。如果效果不好，试试先和并再训练
  
        all_head = nn.Linear(512, num_classes, bias=False) 
        all_head = LogitHead(all_head,logit_scale=4.60517).to(device)  
        all_head.head.weight.data[:lastincclsnum] = best_former_dict.head.weight.data
        all_head.head.weight.data[lastincclsnum:] = logit_head.head.weight.data

    return all_head
    
#new    
def load_data(batch_size, session, args):

    txt_path = "data/index_list/mini_imagenet/session_" + str(session + 1) + '.txt'  #这里要锁定相应的数据路径文件
    index=[]
    lines = [x.strip() for x in open(txt_path, 'r').readlines()]
    for line in lines:
        index.append(line.split('/')[3])
        
    df = {'text': [], 'label':[]}
    
    a=0
    for i in index:
        if a%5 == 0:
            k = int(args.base_class)+(int(session)-1)*10+(int(a)/5)   
            caption = "a logo photo of a "+str(clsname[int(k)])  #要看看下标是否正确对应
            #caption = str(clsname[int(k)])  #要看看下标是否正确对应
            df['text'].append(caption)
            df['label'].append(int(k))
        a=a+1
        
    dataset = Image_caption_dataset(df) 
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    
    return train_dataloader
    
    
def process_text_tuple(text_tuple):
    # 使用 clip.tokenize 处理文本
    tokenized_texts = [clip.tokenize(text) for text in text_tuple]
    
    # 使用 torch.cat 将处理后的文本合并在一起
    concatenated_texts = torch.cat(tokenized_texts)
    
    return concatenated_texts

    
def get_zero_shot_weights(trainloader_t, num_classes, in_features, cmodel, device="cuda"):
    with torch.no_grad():
        weights = torch.zeros(num_classes, in_features)
        textftemp = torch.zeros(num_classes, in_features)
        classindex = []
        for i, batch in enumerate(trainloader_t):  #此时trainloader_t尚未打乱且只有一个batch
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
