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
'76', 'Act II', 'Admiral', 'alpinestars', 'ASUS', 'ATTENIR', 'avery dennison', 'Bacardi', 'battleship', "Bellamy's Australia", 'benuron', 'Bob Evans Restaurants', 'Bold Rock Hard Cider', 'Bovril', 'Brothers Cider', 'Bubba Gump Shrimp Company', 'Bubblicious', 'Bulls-Eye Barbecue', 'Burger King', 'Cafe Coffee Day', 'chatime', 'chewits', 'CoolPAD', 'De Cecco', 'Dongeejiao', 'Enfamil', "Fox's Biscuits", 'g.i. joe', 'Haizhilan', 'Haribo', 'Heineken', 'Jordans', 'Jus-Rol', 'kleenex', 'lexus', 'maybach', 'mclaren', 'new balance', 'new holland', 'playmobil', 'prismacolor', 'prudenial', 'quickchek', 'regina', 'Schiff', 'sherrin', 'shiseido', 'skype', 'staedtler', 'steeden', 'thomapyrin', 'Tim Tam', 'villa zamorano', 'violet crumble', 'vision street wear', 'Vitafusion', 'Wanzaimatou', 'wild berry skittles', 'yorkshire tea', 'zara', 'zendium', 'Taitaile', 'Aveda', 'Aveeno', 'Dr. Oetker', 'La Vie', 'Hancock', 'Impact', 'Sessions', 'Hormel', 'Trust', 'Ace', 'G.A.S', 'IFA', 'The California Raisins', 'JD', 'SPC', 'Crown', 'MiO', 'Heat', 'Angelina', 'Sizzler', 'Orion', 'Pixian', 'Speakeasy', 'Tesla', 'Brunswick', "Mack & Dave's Department Store", 'Westminster', 'Alcon', 'Bona', 'BreadTalk', 'JMP', "Nelsons's", 'RCA', 'Red Barn', 'Sea & Sea', 'KR3W', 'Ola', 'Prodent']


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
        clipcache = torch.load("/home/sdnu2022/桌面/zjx_code/Logo_FSCIL/Logo100_32/mywork/model_logo100_vitb16.pkl", map_location='cuda:0')  # 8.12注释用于对比
        self.cmodel.load_state_dict(clipcache.state_dict()) 
        
        print("-----Text features loading...")
        self.text_features_matrix = torch.zeros(60, 512).to(self.device)  #text features
        for i in range(60):
            texts = clip.tokenize(f"a logo photo of a {clsname[i]}").to(self.device)
            #texts = clip.tokenize(clsname[i]).to(device)
            text_features = self.cmodel.encode_text(texts).float()
            text_features = text_features.flatten()
            self.text_features_matrix[i] = text_features
        print("-----preparation complete!")
        
        self.ocrextracts = []
        #===
        
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
        result_list = [args]

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
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
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
                
                #-------------------new
                inc_num_classes = session * self.args.way
                if session == 1: 
                    #best_head_dict = novelfinetuning(self.cmodel, self.cpreprocess, trainloader, session, args, inc_num_classes)
                    best_head_dict = incfinetune(self.cmodel, self.cpreprocess, trainloader, session, args, inc_num_classes)
                else:
                    #best_head_dict = novelfinetuning(self.cmodel, self.cpreprocess, trainloader, session, args, inc_num_classes, best_head_dict)
                    best_head_dict = incfinetune(self.cmodel, self.cpreprocess, trainloader, session, args, inc_num_classes, best_head_dict)
                best_head_dict = best_head_dict.to(self.device)
                #-------------------

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)
                
                tsl, tsa, ocrextracts = self.test_intergrate(self.model, testloader, 0, args, session, best_head_dict, self.ocrextracts)
                self.ocrextracts = ocrextracts
                
                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                #torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)


    def test_intergrate(self, model, testloader, epoch, args, session, best_head_dict=None, ocrextracts=None, validation=True):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        va = Averager()
        va5= Averager()
        lgt=torch.tensor([])
        lbs=torch.tensor([])
        
        va1 = Averager()
        va2 = Averager()
        va3 = Averager()

        va12 = Averager()
        va123 = Averager()
        va124 = Averager()

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
                #data, test_label = [_.cuda() for _ in batch]
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
                
                logits2 = torch.zeros(batchlen, 100)
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
                        #matcher = SequenceMatcher(None, string1.lower(), string2.lower())  # 创建 SequenceMatcher 实例
                        matcher = SequenceMatcher(None, ocrelems.lower(), string2.lower())  
                        similarity = matcher.ratio()    # 获取相似度百分比
                        logits2[j][k] = similarity
                    
                    #====multimodal similarity calculate for logit3&4
                    image = Image.open(imgpath[j])
                    image_input = self.cpreprocess(image).unsqueeze(0).to(self.device)
                    image_features = self.cmodel.encode_image(image_input).float().to(self.device)
                    base_features_matrix[j] = image_features
                    inc_features_matrix[j] = image_features  

                #logit2 processing     
                for j in range(logits2.shape[0]):
                    row = logits2[j][0:test_class]    # 使用argsort函数对当前行进行排序，并获得排序后的索引
                    sorted_indices = np.argsort(row)  # 使用argsort函数对新数组进行排序，并获得排序后的索引,从小到大
                    row[sorted_indices[-1]] = 0.2
                    row[sorted_indices[-2]] = 0.1
                    row[sorted_indices[-3]] = 0.1
                    row[sorted_indices[:-3]] = 0
                    logits2[j][0:test_class] = row
                logits2 = logits2[:, :test_class].to(self.device) 
                
                #====logits3: base text similarity  
                logits3 = torch.zeros(batchlen, 100)  
                
                clipscore = F.cosine_similarity(base_features_matrix.unsqueeze(1), self.text_features_matrix.unsqueeze(0), dim=2)   #100*100
                clipscore = clipscore.cpu().numpy()
                clipscore = np.log(2*clipscore/ (1 - 2*clipscore))

                logits3[:, :60] = torch.from_numpy(clipscore[:, :60])
                logits3 = logits3[:, :test_class].to(self.device)   
                
                #====logits4: incremental class similarity     
                logits4all = torch.zeros(batchlen, 100)
                logits4temp = best_head_dict(inc_features_matrix)  
                logits4temp = (logits4temp/ (50 - logits4temp))-1  
                #print(logits4temp.shape)    
                logits4all[:, incstart:test_class] = logits4temp  #torch.from_numpy(logits4temp)  
                logits4 = logits4all[:, :test_class].to(self.device)    
                
                #===============================================================

                logits_all = logits + logits2 + 0.7*logits3 + 0.3*logits4     #5*logits3 
                loss = F.cross_entropy(logits_all, test_label)
                acc = count_acc(logits_all, test_label)

                if session > 0:
                    acc1 = count_acc(logits, test_label)
                    acc2 = count_acc(logits2, test_label)
                    acc3 = count_acc(logits3, test_label)
                    
                    acc12 = count_acc(logits+logits2, test_label)
                    acc123 = count_acc(logits+logits2+0.7*logits3, test_label)
                    acc124 = count_acc(logits+logits2+0.3*logits4, test_label)
                    #print(f"class {test_label[0].item()}, acc4: {acc4:.2f}")
                    
                top5acc=count_acc_topk(logits_all, test_label)
                vl.add(loss.item())
                va.add(acc)
                
                va1.add(acc1)
                va2.add(acc2)
                va12.add(acc12)
                
                va3.add(acc3)
                va123.add(acc123)
                
                va124.add(acc124)

                
                va5.add(top5acc)
                lgt=torch.cat([lgt,logits_all.cpu()])
                lbs=torch.cat([lbs,test_label.cpu()])
            vl = vl.item()
            va = va.item()
            
            va1 = va1.item()
            va2 = va2.item()
            va3 = va3.item()

            va12 = va12.item()
            va123 = va123.item()
            va124 = va124.item()
            
            va5= va5.item()
            print('epo {}, test, loss={:.4f} acc={:.4f}, acc1={:.4f}, acc2={:.4f}, acc3={:.4f}'.format(epoch, vl, va, va1, va2,va3))
            print('               acc12={:.4f}, acc123={:.4f}, acc124={:.4f},'.format(va12, va123, va124))
            print("                                                                          ")
            
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm=confmatrix(lgt,lbs,save_model_dir)
            perclassacc=cm.diagonal()
            seenac=np.mean(perclassacc[:args.base_class])
            unseenac=np.mean(perclassacc[args.base_class:])

            
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
