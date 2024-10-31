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
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

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



class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 5
        ctx_init = "a logo photo of a"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors  使用咱给的语句初始化
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized----------

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)


        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        #self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.clip_model = clip_model
        #new
        self.prompted_embedding = None

    def forward(self, image):
        #print(image.shape)  #tensor类型,50*3*84*84
        image_features = self.clip_model.encode_image(image.type(self.dtype))
        
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)    #要让他成为被记录的内容！！！

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        #my new
        self.prompted_embedding = text_features

        return logits

        
class LogitHead(nn.Module):
    def __init__(self, head, device):
        super().__init__()
        self.head = head
        self.textembed = head.weight.data.to(device)
        
    def forward(self, x):
    
        clipscore = F.cosine_similarity(x.unsqueeze(1), self.textembed.unsqueeze(0), dim=2)  
        clipscore = clipscore.cpu().numpy()
        print("============================")
        print(clipscore)
        clipscore = np.log(2*clipscore/ (1 - 2*clipscore))  #batchlen*10
        
        return clipscore

def compute_accuracy(output, target, topk=(1, )):

    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


def novelfinetuning(cmodel, cprocessor, trainloader, session, args, newclsnum, former_head=None):
    
    cmodel.cpu()
    
    max_epoch = 10
    start_index = 120+newclsnum
    end_index = 130+newclsnum
    lastincclsnum = newclsnum-10
    
    print("session:",session,"start:",start_index,"end:",end_index)
    classnames = clsname[start_index:end_index] #获取本次任务的class名称
    print("classnames:",classnames)

    print("Prompt finetuning...")
    cmodel = CustomCLIP(classnames, cmodel)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cmodel.to(device)

    for name, param in cmodel.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    #optimizer = torch.optim.SGD(cmodel.prompt_learner.parameters(),lr=0.002,momentum=0.9,weight_decay=5e-4,dampening=0,nesterov=False)
    optimizer = torch.optim.SGD(cmodel.prompt_learner.parameters(),lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(max_epoch))
    
    #------------微调前精度测试-------------------------
    cmodel.eval()
    progress_bar = tqdm(trainloader, desc=f'Epoch before train')
    for images, labels, paths in progress_bar:
        labels = labels.to(device)
        out_matrix = torch.zeros(50, 10)
        for i in range(len(paths)):
            oneimage = Image.open(paths[i])
            oneimage = cprocessor(oneimage).unsqueeze(0).to(device)
            output = cmodel(oneimage).float() 
            out_matrix[i] = output

        out_matrix = out_matrix.to(device)
        mapped_labels = labels.clone()  
        for key, value in incmapping.items():
            mapped_labels[labels == key] = value
        acc = compute_accuracy(out_matrix, mapped_labels)[0].item(),
        print('Befores acc:',acc[0])
    
    #-----------------正式微调--------------------------------------
    best_acc = 0
    best_embedding = torch.zeros(10, 512)
    
    linear_head = nn.Linear(512, 10, bias=False) 
    new_head = LogitHead(linear_head,device).train().to(device)   #这个logit_scale加不加效果会有影响吗
    
    for epoch in range(max_epoch):
        cmodel.train()
        avg_loss = AvgMeter()
        progress_bar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{10}')
        for images, labels, paths in progress_bar:
            #images = images.to(CFG.device)
            labels = labels.to(device)
            out_matrix = torch.zeros(50, 10)
            
            for i in range(len(paths)):
                oneimage = Image.open(paths[i])
                oneimage = cprocessor(oneimage).unsqueeze(0).to(device)
                output = cmodel(oneimage).float() 
                out_matrix[i] = output

            out_matrix = out_matrix.to(device)

            mapped_labels = labels.clone()  
            for key, value in incmapping.items():
                mapped_labels[labels == key] = value
            
            train_acc = compute_accuracy(out_matrix, mapped_labels)[0].item(),
            loss = F.cross_entropy(out_matrix, mapped_labels)
            #print('acc:',train_acc[0])
            
            #new
            if train_acc[0] > best_acc:
                best_acc = train_acc[0]
                print("*****get new prompt embedding!*****")
                text_features = cmodel.prompted_embedding  #--------------------------
                new_head.head.weight.data = text_features

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss.update(loss.item())
            progress_bar.set_postfix({'loss': avg_loss.avg})

        progress_bar.close()
        print(f'Epoch:{epoch+1}, Loss:{avg_loss.avg:.5f}, acc:{train_acc[0]}')

    all_head = new_head	 
    if session>1:                                                                                 #进行权重拼接:每一步的prompt一次又一次拼接
        all_head = nn.Linear(512, newclsnum, bias=False) 
        all_head = LogitHead(all_head).to(device)  
        all_head.head.weight.data[:lastincclsnum] = former_head.head.weight.data
        all_head.head.weight.data[lastincclsnum:] = new_head.head.weight.data

    print(f"Session {session} finetuning finish")
    
    return all_head
    

    

