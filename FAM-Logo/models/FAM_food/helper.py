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
'Perdue', 'Asahi', 'Appleton Estate', 'Horlicks', 'Heinz Baked Beans', 'Baja Fresh', "Lay's", 'waffle crisp', 'Burger King', 'yili', 'Harvest Crunch', 'mccafe', 'evian', 'peptamen', "Newman's Own", "Gregg's", 'second cup', 'jack in the box', 'Chocapic', 'winiary', 'Maestro Dobel', 'Chocolate Lucky Charms', 'molson brador', 'Brothers Cider', 'Boca Burger', 'sweetened wheat-fuls', 'Highland Spring', 'Hard Rock Cafe', 'Cascadian Farm', 'Maruchan', 'Cibo Espresso', 'Cafe du Monde', 'Michel et Augustin', 'Beer Nuts', 'Cocoa Puffs', 'Mecca Cola', "Bush's Best", 'Carling Black Label', 'vladivar', 'Straus', 'Four Seas Ice Cream', "McDonald's", 'stadium mustard', 'Mondelez International', 'Baxters', 'xifeng', 'Bahlsen', 'robust', 'Celestial Seasonings', 'Britannia', 'chicago town', 'Hortex', 'BreadTalk', 'nesvita', 'CHIVAS', 'Alter Eco', 'Darden', "Brigham's Ice Cream", "Maker's Mark", 'Hormel', "Eegee's", 'Hollys Coffee', 'copella', 'strawberry rice krispies', 'Frosty Boy', 'lan-choo', 'Gobstoppers', 'Burgerville', 'Bellywashers', 'Mengzhilan', "Mike's", 'Bon Pari', 'FruChocs', 'Cocoa Pebbles', 'Bicerin', 'Tyrrells', "tully's coffee", 'Aqua Carpatica', "Fox's Pizza Den", 'Honey Bunches of Oats', 'Divine', 'Cracker', 'sveltesse', "Kwality Wall's", 'laura secord chocolates', 'Efes', 'Canada Dry', 'ready brek', 'Buondi', 'nutren', 'La Choy', 'Fruco', 'Heinz Tomato Ketchup', 'Ketel One', "romano's macaroni grill", 'ricoffy', 'Ciego Montero', 'Irn Bru Bar', 'Cestbon', 'Cafe HAG', 'Meiji', 'Kotipizza', 'Cherry 7Up', 'Badia', 'Johnny Rockets', 'Huy Fong Foods', 'Balaji Wafers', 'Apple Zings', "Bewley's", 'Carling', 'Bigg Mixx', 'Coco Pops', 'APEROL', 'Jelly Belly', 'Calbee', 'Cielo', 'Bridgehead Coffee', 'lion cereal', 'Chipsmore', 'Grapette', 'toblerone', 'Bartolo Nardini', 'smirnoff', 'Burger Street', 'wuyutai', "rosati's", 'Cha Dao', "Fox's Biscuits", 'chex mix', 'Poulain', 'Laciate', 'Danone', 'Dalda', 'ledo pizza', 'Maggi Masala noodles', 'CDO Foodsphere', 'Highland Toffee', 'Goya', 'Idaho SPUD', 'screaming yellow zonkers', 'maggi noodles', 'Gay Lea', 'Cocosette', 'Bisquick', "ching's secret", 'Frosty Jack Cider', 'NISSIN', 'Monsanto', 'Chocomel', "Haldiram's", "Graeter's", 'nestle', 'tooty footies', 'oberweis dairy', "Hershey's Cookies 'n' Creme", "timothy's world coffee", 'williams fresh cafe', 'Coco Roos', 'COFCO', 'DEVONDALE', 'Gordon Food Service', 'Cape Cod Potato Chips', "Bruegger's", 'nestle corn flakes', 'Carapelli', 'Greene King', "Lady's Choice", 'Freihofers', 'la saltena', 'Frosted Shredded Wheat', 'Brugal', 'Chips Ahoy', 'Iams', 'Fujiya', 'Buca di Beppo', 'Qianhe', 'Creemore', 'obaras', 'st arnou', 'Hot Pockets', 'Laffy Taffy', 'pisco porton', 'Chronic Tacos', 'Haribo', 'Frosted Mini Spooners', 'Juicy Fruit', 'Beaulieu', 'Maltesers', 'Polar', 'Eden Cheese', 'Boon Rawd Brewery', 'Magners lrish', 'Jollibee', 'Chef Boyardee', 'robeks', "Kern's", 'Coors', "steve's ice cream", 'Capri Sun', 'Bega']

def base_train(model, trainloader, optimizer, scheduler, epoch, args,mask):
    tl = Averager()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):

        beta=torch.distributions.beta.Beta(args.alpha, args.alpha).sample([]).item()  
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
            
            logits2 = torch.zeros((batchlen, 200))  
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
                    matcher = SequenceMatcher(None, string1.lower(), string2.lower())  
                    similarity = matcher.ratio()    
                    logits2[j][k] = similarity
                
                #====multimodal similarity calculate for logit3
                image = Image.open(imgpath[j])
                image_input = cprocess(image).unsqueeze(0).to(device)
                image_features = cmodel.encode_image(image_input).float().to(device)
                base_features_matrix[j] = image_features
                inc_features_matrix[j] = image_features  

            #logit2 processing     
            for j in range(logits2.shape[0]):
                row = logits2[j][0:test_class]    
                sorted_indices = np.argsort(row)  
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

            logits3 = torch.zeros((batchlen, 200))    
            logits3[:, :100] = torch.from_numpy(clipscore[:, :100])
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
        print('epo {}, test, loss={:.4f}, acc={:.4f}, acc1={:.4f}, acc2={:.4f}, acc3={:.4f}'.format(epoch, vl, va,acc1,acc2,acc3))
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
