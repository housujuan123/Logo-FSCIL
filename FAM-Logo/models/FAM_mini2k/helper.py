# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

import clip
import easyocr
from difflib import SequenceMatcher
from PIL import Image


clsname=[
'24seven', 'A1 Steak Sauce', 'ABLE', 'Abuelita', 'aceh', 'Addidas', 'Aero Contractors', 'Algida', 'Alpen Gold', 'Amkette', 'Ampeg', "Amy's Ice Creams", 'Apple Jacks', 'Aprilia', 'Aquafresh', 'ASCASO', 'Ashbury', 'Astro', 'Asusu', 'Atlantic Airlines', 'Backwoods', 'BAE', 'Bahia', 'Baker Skateboards', 'Bakers', 'Banana Frosted Flakes', 'Baraka', 'Batman', 'Billabong', 'Blackthorn Cider', 'Blaser', 'Blue Bonnet', 'Blue Stratos', "Boscov's", 'Brazil', 'Bridgestone', 'Bristol', 'Brooklyn', 'Buffalo', 'Buffalo Rock Ginger Ale', "Caffrey's", 'Cailler', 'Cain', 'Caloi', 'Cape Cod Potato Chips', 'Caress', 'Carrefour', 'Carte Noire', 'Centurion', 'cestbon', 'CHEF', 'Chery', 'Chiclets', "Chili's", 'CHINA CITIC', 'Chocapic', 'CHS', "Chuck E. Cheese's", 'Cigar City', 'Circa', 'Clinic', 'Coco Pops', 'Coffee-Mate', 'Connaught', 'Connector', 'Contour', 'Cool Whip', 'Diet Coke', 'Dimension', 'Dollar', "Domino's Pizza", 'Double Take', "Dr Brown's", 'DUX', 'DVS Shoes', 'Eclipse', 'El Jimador', 'Ellesse', 'eMachines', 'Escada', 'Eureka', 'Expert', 'Fram', 'Frito-Lay', 'Fujitsu', 'Full Moon', 'Geely', 'Gel', 'Gillette', 'Glock', 'GoPro', 'Granada', 'Granville', 'Gull', 'Guojiao', "Hartley's", 'Heartland', 'Highland Toffee', 'Honey Kix', 'Honey Nut Cheerios', "Howard Johnson's", 'Hummel', 'Husqvarna', 'Ibex', 'Ilford Photo', 'Intermarche', 'Irn Bru Bar', 'ITS', 'Ivory', "Jason's Deli", 'KEF', 'Kingdom', 'Kodak', 'Koenigsegg', 'Koko Krunch', 'Kopparbergs', 'Krush', 'KTM', 'Lacto', 'Lada', 'Laffy Taffy', 'Lapierre', 'Lean Cuisine', 'LEE KUM KEE', 'Lefevre-Utile', 'Lego', 'Lemonade', 'Limca', 'Liptonice', "LLoyd's", 'logo Bubblicious', 'Lotus Cars', 'Love is', 'Loyd', 'Lynx', 'MAD DOG', 'Magners Irish', 'Maple', 'Mattel', 'Maxum', 'Maxwell House', 'Mazda', 'Metz', 'Mickey Mouse', 'Mighty Dog', 'MiO', 'Mirage', 'Mirinda', 'Mitsubishi', 'Monster Munch', 'Mr Kipling', 'Mrs. Fields', 'Multi-Bran Chex', 'Murphy', 'Naleczowianka', 'NAN HA', 'Nerds', 'Nesfruta', 'Nestum', "Nice 'n Easy", 'North Coast', 'Nostromo', "O'Charley's", 'Orange Blossom', 'Orangina', 'Orbit', 'Origen', 'Orion', 'Oronamin C', 'Overture', 'Pathe', 'PDV', 'Pechoin', "Peet's Coffee & Tea", 'PEL', 'pepsi', 'Petro', 'Petronas', 'Philips', 'Pioneer', 'Pixian', 'Primos', 'ProTech', 'Proto', 'Quasar', 'Realtor', 'Red Barn', 'Red Hot & Blue', 'Red Oak', 'REFOSIAN', 'Refreshers Gums', 'Revolution', 'Reynolds', "Richard Petty 43's", 'Rip Curl', 'Russell', 'Saladworks', 'Sbarro', 'Screamers', 'Sharp', 'Solex', 'sony', 'Spider-Man', 'SR', 'Stanley', 'Star Energy', 'Steam Whistle', 'Stella', 'Sure', 'Tatiana', 'TECO', 'The Batman', 'thirst cola', 'Toby', 'Toscano', 'Total', 'Transcom', 'Trek Bicycle', 'Triaminic', 'Trio', 'Twist', 'Tylenol', 'Valero', 'Vegemite', 'Vestel', 'wanglaoji', 'Wegmans', 'Williams', 'WMF', 'Xurisheng']

def base_train(model, trainloader, optimizer, scheduler, epoch, args,mask):
    tl = Averager()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):

        beta=torch.distributions.beta.Beta(args.alpha, args.alpha).sample([]).item()
        #data, train_label = [_.cuda() for _ in batch]
        data, train_label, imgpath = batch[0].cuda(), batch[1].cuda(), batch[2]
        
        embeddings=model.module.encode(data)

        logits = model(data)
        logits_ = logits[:, :args.base_class]
        loss = F.cross_entropy(logits_, train_label)
        
        acc = count_acc(logits_, train_label)
        
        
        if epoch>=args.loss_iter:
            logits_masked = logits.masked_fill(F.one_hot(train_label, num_classes=model.module.pre_allocate) == 1, -1e9)
            logits_masked_chosen= logits_masked * mask[train_label]
            pseudo_label = torch.argmax(logits_masked_chosen[:,args.base_class:], dim=-1) + args.base_class
            #pseudo_label = torch.argmax(logits_masked[:,args.base_class:], dim=-1) + args.base_class
            loss2 = F.cross_entropy(logits_masked, pseudo_label)

            index = torch.randperm(data.size(0)).cuda()
            pre_emb1=model.module.pre_encode(data)
            mixed_data=beta*pre_emb1+(1-beta)*pre_emb1[index]
            mixed_logits=model.module.post_encode(mixed_data)

            newys=train_label[index]
            idx_chosen=newys!=train_label
            mixed_logits=mixed_logits[idx_chosen]

            pseudo_label1 = torch.argmax(mixed_logits[:,args.base_class:], dim=-1) + args.base_class # new class label
            pseudo_label2 = torch.argmax(mixed_logits[:,:args.base_class], dim=-1)  # old class label
            loss3 = F.cross_entropy(mixed_logits, pseudo_label1)
            novel_logits_masked = mixed_logits.masked_fill(F.one_hot(pseudo_label1, num_classes=model.module.pre_allocate) == 1, -1e9)
            loss4 = F.cross_entropy(novel_logits_masked, pseudo_label2)
            total_loss = loss+args.balance*(loss2+loss3+loss4)
        else:
            total_loss = loss


        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        #loss.backward()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            #data, label = [_.cuda() for _ in batch]
            data, label, imgpath = batch[0].cuda(), batch[1].cuda(), batch[2]
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        #data_index = (label_list == class_index).nonzero()
        data_index = torch.nonzero((label_list == class_index))
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list

    return model



def test(model, testloader, epoch,args, session,validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    lgt=torch.tensor([])
    lbs=torch.tensor([])
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            #data, test_label = [_.cuda() for _ in batch]
            data, test_label, _ = batch[0].cuda(), batch[1].cuda(), batch[2]
            logits = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            vl.add(loss.item())
            va.add(acc)
            lgt=torch.cat([lgt,logits.cpu()])
            lbs=torch.cat([lbs,test_label.cpu()])
        vl = vl.item()
        va = va.item()
        print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        
        lgt=lgt.view(-1,test_class)
        lbs=lbs.view(-1)
        if validation is not True:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm=confmatrix(lgt,lbs,save_model_dir)
            perclassacc=cm.diagonal()
            seenac=np.mean(perclassacc[:args.base_class])
            unseenac=np.mean(perclassacc[args.base_class:])
            print('Seen Acc:',seenac, 'Unseen ACC:', unseenac)
    return vl, va

def test_cross(model, testloader, epoch,args, session, text_features_matrix, cmodel, cprocess, reader, ocrextracts, validation=True):
    test_class = args.base_class + session * args.way
    
    model = model.eval()
    cmodel = cmodel.eval()
    vl = Averager()
    va = Averager()
    lgt=torch.tensor([])
    lbs=torch.tensor([])
    
    va1 = Averager()
    va2 = Averager()
    va3 = Averager()
    va12 = Averager()
    va123 = Averager()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    with torch.no_grad():
        samplenum = 0                                                                             #-------       
        for i, batch in enumerate(testloader, 1):
            #data, test_label = [_.cuda() for _ in batch]
            data, test_label, imgpath = batch[0].cuda(), batch[1].cuda(), batch[2]
            
            logits = model(data)
            logits = logits[:, :test_class]
            logits = 0.05*logits
            
            #=====================New====================================
            batchlen = len(imgpath) 
            base_features_matrix = torch.zeros(batchlen, 512).to(device)  
            inc_features_matrix = torch.zeros(batchlen, 512).to(device)  
            
            logits2 = torch.zeros((batchlen, 230))  
            for j in range(batchlen):    # one by one
                #====logits2: OCR
                try:
                    results = reader.readtext(imgpath[j], detail = 0)
                    
                except Exception as e:
                    print("An error occurred:", e) 
                    
                string1 = ""
                for k in range(len(results)):
                    string1 = string1+results[k]
                ocrextracts.append(string1)
                    
                for k in range(test_class):
                    string2 = clsname[k]
                    matcher = SequenceMatcher(None, string1.lower(), string2.lower())  # 创建 SequenceMatcher 实例
                    similarity = matcher.ratio()    # 获取相似度百分比
                    logits2[j][k] = similarity
                
                #====multimodal similarity calculate for logit3
                image = Image.open(imgpath[j])
                image_input = cprocess(image).unsqueeze(0).to(device)
                image_features = cmodel.encode_image(image_input).float().to(device)
                base_features_matrix[j] = image_features
                inc_features_matrix[j] = image_features  

            #logit2 processing     
            for j in range(logits2.shape[0]):
                row = logits2[j][0:test_class]    # 使用argsort函数对当前行进行排序，并获得排序后的索引
                sorted_indices = np.argsort(row)  # 使用argsort函数对新数组进行排序，并获得排序后的索引,从小到大
                row[sorted_indices[-1]] = 0.3
                row[sorted_indices[-2]] = 0.1
                row[sorted_indices[-3]] = 0.1
                row[sorted_indices[:-3]] = 0
                logits2[j][0:test_class] = row
            logits2 = logits2[:, :test_class].to(device) 
            
            #====logits3: base text similarity  
            clipscore = F.cosine_similarity(base_features_matrix.unsqueeze(1), text_features_matrix.unsqueeze(0), dim=2)   #100*130
            clipscore = clipscore.cpu().numpy()
            clipscore = np.log(2*clipscore/ (1 - 2*clipscore))

            logits3 = torch.zeros((batchlen, 230))    
            logits3[:, :130] = torch.from_numpy(clipscore[:, :130])
            logits3 = logits3[:, :test_class].to(device)
            logits3 = 0.5*logits3   

            #===============================================================
            logits_all = logits + logits2 + 0.45*logits3
            loss = F.cross_entropy(logits_all, test_label)
            acc = count_acc(logits_all, test_label)
             
            if session == 0 :
                acc1 = count_acc(logits, test_label)
                acc2 = count_acc(logits2, test_label)
                acc3 = count_acc(logits3, test_label)
                #print(f"class {test_label[0].item()}, accall: {acc:.2f}, acc1: {acc1:.2f}, acc2: {acc2:.2f}, acc3: {acc3:.2f}")
                acc12 = count_acc(logits+logits2, test_label)
                acc123 = acc
            
            vl.add(loss.item())
            va.add(acc)
            
            va1.add(acc1)
            va2.add(acc2)
            va3.add(acc3)
            va12.add(acc12)
            va123.add(acc123)
                
            lgt=torch.cat([lgt,logits_all.cpu()])
            lbs=torch.cat([lbs,test_label.cpu()])
        
        va1 = va1.item()
        va2 = va2.item()
        va3 = va3.item()
        va12 = va12.item()
        va123 = va123.item()
        vl = vl.item()
        va = va.item()
        print('epo {}, test, loss={:.4f}, acc={:.4f}, acc1={:.4f}, acc2={:.4f}, acc3={:.4f}'.format(epoch, vl, va,va1,va2,va3))
        print('              acc12={:.4f}, acc123={:.4f}'.format(va12, va123))
        print("                                                                          ")

        
        lgt=lgt.view(-1,test_class)
        lbs=lbs.view(-1)
        if validation is not True:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm=confmatrix(lgt,lbs,save_model_dir)
            perclassacc=cm.diagonal()
            seenac=np.mean(perclassacc[:args.base_class])
            unseenac=np.mean(perclassacc[args.base_class:])
            print('Seen Acc:',seenac, 'Unseen ACC:', unseenac)
    return vl, va, ocrextracts


def test_withfc(model, testloader, epoch,args, session,validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    lgt=torch.tensor([])
    lbs=torch.tensor([])
    with torch.no_grad(): 
        for i, batch in enumerate(testloader, 1):
            #data, test_label = [_.cuda() for _ in batch]
            data, test_label, _ = batch[0].cuda(), batch[1].cuda(), batch[2]
            logits = model.module.forpass_fc(data)
            logits = logits[:, :test_class]

            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)

            vl.add(loss.item())
            va.add(acc)
            lgt=torch.cat([lgt,logits.cpu()])
            lbs=torch.cat([lbs,test_label.cpu()])
        vl = vl.item()
        va = va.item()
        print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        
        lgt=lgt.view(-1,test_class)
        lbs=lbs.view(-1)
        if validation is not True:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm=confmatrix(lgt,lbs,save_model_dir)
            perclassacc=cm.diagonal()
            seenac=np.mean(perclassacc[:args.base_class])
            unseenac=np.mean(perclassacc[args.base_class:])
            print('Seen Acc:',seenac, 'Unseen ACC:', unseenac)
    return vl, va
