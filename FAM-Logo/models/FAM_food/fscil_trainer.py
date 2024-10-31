from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *

import clip
import easyocr
from difflib import SequenceMatcher
from .incfinetuning import *

clsname=[
'Perdue', 'Asahi', 'Appleton Estate', 'Horlicks', 'Heinz Baked Beans', 'Baja Fresh', "Lay's", 'waffle crisp', 'Burger King', 'yili', 'Harvest Crunch', 'mccafe', 'evian', 'peptamen', "Newman's Own", "Gregg's", 'second cup', 'jack in the box', 'Chocapic', 'winiary', 'Maestro Dobel', 'Chocolate Lucky Charms', 'molson brador', 'Brothers Cider', 'Boca Burger', 'sweetened wheat-fuls', 'Highland Spring', 'Hard Rock Cafe', 'Cascadian Farm', 'Maruchan', 'Cibo Espresso', 'Cafe du Monde', 'Michel et Augustin', 'Beer Nuts', 'Cocoa Puffs', 'Mecca Cola', "Bush's Best", 'Carling Black Label', 'vladivar', 'Straus', 'Four Seas Ice Cream', "McDonald's", 'stadium mustard', 'Mondelez International', 'Baxters', 'xifeng', 'Bahlsen', 'robust', 'Celestial Seasonings', 'Britannia', 'chicago town', 'Hortex', 'BreadTalk', 'nesvita', 'CHIVAS', 'Alter Eco', 'Darden', "Brigham's Ice Cream", "Maker's Mark", 'Hormel', "Eegee's", 'Hollys Coffee', 'copella', 'strawberry rice krispies', 'Frosty Boy', 'lan-choo', 'Gobstoppers', 'Burgerville', 'Bellywashers', 'Mengzhilan', "Mike's", 'Bon Pari', 'FruChocs', 'Cocoa Pebbles', 'Bicerin', 'Tyrrells', "tully's coffee", 'Aqua Carpatica', "Fox's Pizza Den", 'Honey Bunches of Oats', 'Divine', 'Cracker', 'sveltesse', "Kwality Wall's", 'laura secord chocolates', 'Efes', 'Canada Dry', 'ready brek', 'Buondi', 'nutren', 'La Choy', 'Fruco', 'Heinz Tomato Ketchup', 'Ketel One', "romano's macaroni grill", 'ricoffy', 'Ciego Montero', 'Irn Bru Bar', 'Cestbon', 'Cafe HAG', 'Meiji', 'Kotipizza', 'Cherry 7Up', 'Badia', 'Johnny Rockets', 'Huy Fong Foods', 'Balaji Wafers', 'Apple Zings', "Bewley's", 'Carling', 'Bigg Mixx', 'Coco Pops', 'APEROL', 'Jelly Belly', 'Calbee', 'Cielo', 'Bridgehead Coffee', 'lion cereal', 'Chipsmore', 'Grapette', 'toblerone', 'Bartolo Nardini', 'smirnoff', 'Burger Street', 'wuyutai', "rosati's", 'Cha Dao', "Fox's Biscuits", 'chex mix', 'Poulain', 'Laciate', 'Danone', 'Dalda', 'ledo pizza', 'Maggi Masala noodles', 'CDO Foodsphere', 'Highland Toffee', 'Goya', 'Idaho SPUD', 'screaming yellow zonkers', 'maggi noodles', 'Gay Lea', 'Cocosette', 'Bisquick', "ching's secret", 'Frosty Jack Cider', 'NISSIN', 'Monsanto', 'Chocomel', "Haldiram's", "Graeter's", 'nestle', 'tooty footies', 'oberweis dairy', "Hershey's Cookies 'n' Creme", "timothy's world coffee", 'williams fresh cafe', 'Coco Roos', 'COFCO', 'DEVONDALE', 'Gordon Food Service', 'Cape Cod Potato Chips', "Bruegger's", 'nestle corn flakes', 'Carapelli', 'Greene King', "Lady's Choice", 'Freihofers', 'la saltena', 'Frosted Shredded Wheat', 'Brugal', 'Chips Ahoy', 'Iams', 'Fujiya', 'Buca di Beppo', 'Qianhe', 'Creemore', 'obaras', 'st arnou', 'Hot Pockets', 'Laffy Taffy', 'pisco porton', 'Chronic Tacos', 'Haribo', 'Frosted Mini Spooners', 'Juicy Fruit', 'Beaulieu', 'Maltesers', 'Polar', 'Eden Cheese', 'Boon Rawd Brewery', 'Magners lrish', 'Jollibee', 'Chef Boyardee', 'robeks', "Kern's", 'Coors', "steve's ice cream", 'Capri Sun', 'Bega']


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()
        
        #==new
        self.device = "cuda" if torch.cuda.is_available() else "cpu"    
        print("-----OCR reader loading...")
        self.reader = easyocr.Reader(['en'], gpu = True)
        
        print("-----CLIP model loading...")
        self.cmodel, self.cpreprocess = clip.load("ViT-B/16", device=self.device, jit=False)
        clipcache = torch.load("/home/sdnu2022/桌面/zjx_code/Logo_FSCIL/foodlogo200/mywork/model_food200_vitb16.pkl")  
        self.cmodel.load_state_dict(clipcache.state_dict()) 
        
        print("-----Text features loading...")
        self.text_features_matrix = torch.zeros(100, 512).to(self.device)  #text features
        for i in range(100):
            texts = clip.tokenize(f"a logo photo of a {clsname[i]}").to(self.device)
            text_features = self.cmodel.encode_text(texts).float()
            text_features = text_features.flatten()
            self.text_features_matrix[i] = text_features
        print("-----preparation complete!")
        
        self.ocrextracts = []

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    
    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        self.result_list = [args]

        #gen_mask
        masknum=3
        mask=np.zeros((args.base_class,args.num_classes))
        for i in range(args.num_classes-args.base_class):
            picked_dummy=np.random.choice(args.base_class,masknum,replace=False)
            mask[:,i+args.base_class][picked_dummy]=1
        mask=torch.tensor(mask).cuda()

        
        for session in range(args.start_session, args.sessions):
            train_set, trainloader, testloader = self.get_dataloader(session)
            self.model.load_state_dict(self.best_model_dict)
            
            if session == 0:  # load base class train img label
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args,mask)
                    # test model with all seen class
                    tsl, tsa = test(self.model, testloader, epoch, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    self.result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                self.result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'
                    tsl, tsa, ocrextracts = test_cross(self.model, testloader, 0, args, session, self.text_features_matrix, self.cmodel, self.cpreprocess, self.reader, self.ocrextracts)   #new--------------
                    self.ocrextracts = ocrextracts
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

                #save dummy classifiers
                self.dummy_classifiers=deepcopy(self.model.module.fc.weight.detach())
                
                self.dummy_classifiers=F.normalize(self.dummy_classifiers[self.args.base_class:,:],p=2,dim=-1)
                self.old_classifiers=self.dummy_classifiers[:self.args.base_class,:]

            else:  # incremental learning sessions
                print("training session: [%d]" % session)
                
                inc_num_classes = session * self.args.way
                if session == 1: 
                    best_head_dict = incfinetune(self.cmodel, self.cpreprocess, trainloader, session, args, inc_num_classes)
                else:
                    best_head_dict = incfinetune(self.cmodel, self.cpreprocess, trainloader, session, args, inc_num_classes, best_head_dict)
                best_head_dict = best_head_dict.to(self.device)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)
                
                tsl, tsa, ocrextracts = self.test_intergrate(self.model, testloader, 0, args, session, best_head_dict, self.ocrextracts)
                self.ocrextracts = ocrextracts
                
                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                self.result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        self.result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        self.result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), self.result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)


    def test_intergrate(self, model, testloader, epoch, args, session, best_head_dict=None, ocrextracts=None, validation=True):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        va = Averager()
        lgt=torch.tensor([])
        lbs=torch.tensor([])
        proj_matrix=torch.mm(self.dummy_classifiers,F.normalize(torch.transpose(model.module.fc.weight[:test_class, :],1,0),p=2,dim=-1))
        eta=args.eta
        softmaxed_proj_matrix=F.softmax(proj_matrix,dim=1)
        inc_class = session * args.way
        incstart = args.base_class
        ocrindex = 0
        with torch.no_grad():
            if session>0:  
                best_head_dict.eval()
            for i, batch in enumerate(testloader, 1):
                data, test_label, imgpath = batch[0].cuda(), batch[1].cuda(), batch[2]
                emb=model.module.encode(data)
                proj=torch.mm(F.normalize(emb,p=2,dim=-1),torch.transpose(self.dummy_classifiers,1,0))
                topk, indices = torch.topk(proj, 40)
                res = (torch.zeros_like(proj))
                res_logit = res.scatter(1, indices, topk)
                logitsa=torch.mm(res_logit,proj_matrix)
                logitsb = model.module.forpass_fc(data)[:, :test_class] 
                logits=eta*F.softmax(logitsa,dim=1)+(1-eta)*F.softmax(logitsb,dim=1)

                #=====================New====================================
                batchlen = len(imgpath) 
                base_features_matrix = torch.zeros(batchlen, 512).to(self.device)  
                inc_features_matrix = torch.zeros(batchlen, 512).to(self.device)  
                logits2 = torch.zeros(batchlen, 200)
                for j in range(batchlen):    # one by one
                    #====logits2: OCR
                    if ocrindex < len(ocrextracts):
                        ocrelems = ocrextracts[ocrindex]
                        ocrindex = ocrindex+1
                    else:
                        try:
                            results = self.reader.readtext(imgpath[j], detail = 0)
                        except Exception as e:
                            print("An error occurred:", e) 
                        string1 = ""
                        for k in range(len(results)):
                            string1 = string1+results[k]  
                        ocrelems = string1
                        ocrextracts.append(ocrelems) 
                    for k in range(test_class):
                        string2 = clsname[k]
                        matcher = SequenceMatcher(None, ocrelems.lower(), string2.lower())  
                        similarity = matcher.ratio()    
                        logits2[j][k] = similarity
                    
                    #====multimodal similarity calculate for logit3&4
                    image = Image.open(imgpath[j])
                    image_input = self.cpreprocess(image).unsqueeze(0).to(self.device)
                    image_features = self.cmodel.encode_image(image_input).float().to(self.device)
                    base_features_matrix[j] = image_features
                    inc_features_matrix[j] = image_features  

                #logit2 processing     
                for j in range(logits2.shape[0]):
                    row = logits2[j][0:test_class]    
                    sorted_indices = np.argsort(row)  
                    row[sorted_indices[-1]] = 0.2
                    row[sorted_indices[-2]] = 0.1
                    row[sorted_indices[-3]] = 0.1
                    row[sorted_indices[:-3]] = 0
                    logits2[j][0:test_class] = row
                logits2 = logits2[:, :test_class].to(self.device) 
                
                #====logits3: base text similarity  
                logits3 = torch.zeros(batchlen, 200)  
                clipscore = F.cosine_similarity(base_features_matrix.unsqueeze(1), self.text_features_matrix.unsqueeze(0), dim=2)  
                clipscore = clipscore.cpu().numpy()
                clipscore = np.log(2*clipscore/ (1 - 2*clipscore))
                logits3[:, :100] = torch.from_numpy(clipscore[:, :100])
                logits3 = logits3[:, :test_class].to(self.device)   
                
                #====logits4: incremental class similarity     
                logits4all = torch.zeros(batchlen, 200)
                logits4temp = best_head_dict(inc_features_matrix)  
                logits4temp = (logits4temp/ (50 - logits4temp))-1      
                logits4all[:, incstart:test_class] = logits4temp 
                logits4 = logits4all[:, :test_class].to(self.device)    
                
                #===============================================================
                logits_all = logits + logits2 + 0.45*logits3 + 0.15*logits4     
                loss = F.cross_entropy(logits_all, test_label)
                acc = count_acc(logits_all, test_label)
                    
                top5acc=count_acc_topk(logits_all, test_label)
                vl.add(loss.item())
                va.add(acc)
                lgt=torch.cat([lgt,logits_all.cpu()])
                lbs=torch.cat([lbs,test_label.cpu()])
            vl = vl.item()
            va = va.item()
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm=confmatrix(lgt,lbs,save_model_dir)
            perclassacc=cm.diagonal()
            seenac=np.mean(perclassacc[:args.base_class])
            unseenac=np.mean(perclassacc[args.base_class:])
            self.result_list.append('Seen Acc:%.5f, Unseen ACC:%.5f' % (seenac,unseenac))

            
        return vl, va, ocrextracts

    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
            self.args.save_path = self.args.save_path + 'Bal%.2f-LossIter%d' % (
                self.args.balance, self.args.loss_iter)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + 'Cosine-Epo_%d-Lr_%.4f' % (
                self.args.epochs_base, self.args.lr_base)
            self.args.save_path = self.args.save_path + 'Bal%.2f-LossIter%d' % (
                self.args.balance, self.args.loss_iter)

        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
