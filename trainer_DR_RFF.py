import marveltoolbox as mt 
from src.models import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from src.dataset import *
from src.evaluation import *
from src.Unet import SUNet, SUNetZ


class Confs(mt.BaseConfs):
    def __init__(self, train_snr,
     device=0, d1=None, d2=None, z_dim=32, lamda=1.0, alpha=1.0, beta=1.0, epsilon=0.0, is_NS=False, is_HP=True, is_swap=True, is_FIR=False, is_TS=False):
        self.train_snr = train_snr
        self.device = device
        self.device_ids = [device]
        self.d1 = d1
        self.d2 = d2
        self.z_dim = z_dim
        self.lamda = lamda
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.is_NS = is_NS
        self.is_HP = is_HP
        self.is_FIR = is_FIR
        self.is_TS = is_TS
        self.is_swap = is_swap
        super().__init__()
        
    
    def get_dataset(self):
        self.dataset = 'val'
        self.nc = 2
        self.batch_size = 256
        self.class_num = 54
        self.epochs = 50
        self.max_iter = 10
        self.seed = 0

    def get_flag(self):
        if self.is_TS:
            self.data_idx = -1
        else:
            self.data_idx = 0

        self.flag = 'DR-RFF-{}-snr{}-nz{}-L{}-lamda{}-alpha{}-beta{}-eps{}'.format(
            self.dataset, self.train_snr, self.z_dim, 
            self.d2, self.lamda, self.alpha, self.beta, self.epsilon)
        if self.is_NS:
            self.flag += '-NS'
        if self.is_TS:
            self.flag += '-TS'
        if not self.is_HP:
            self.flag += '-noHP'
        if self.is_FIR:
            self.flag += '-FIR'
        if not self.is_swap:
            self.flag += '-noswap'
        

    def get_device(self):
        self.device = torch.device(
            "cuda:{}".format(self.device) if \
            torch.cuda.is_available() else "cpu")


class Trainer(mt.BaseTrainer, Confs):
    def __init__(
        self, train_snr=None, device=3, d1=None, d2=16, z_dim=512, 
        lamda=1.0, alpha=1.0, beta=1.0, epsilon=0.0, is_NS=False, is_HP=True, is_swap=True, is_FIR=False, is_TS=False):

        Confs.__init__(
            self, train_snr, device, d1, d2, z_dim, 
            lamda, alpha, beta, epsilon, is_NS, is_HP, is_swap, is_FIR, is_TS)
        mt.BaseTrainer.__init__(self, self)

        if self.is_HP and self.is_NS:
            self.models['F'] = NS_CLF_L2Softmax(in_channels=self.nc, out_channels=self.class_num, d1=8, d2=self.d2-2, z_dim=self.z_dim).to(self.device)
        elif (not self.is_HP) and self.is_NS:
            self.models['F'] = NS_CLF_Softmax(in_channels=self.nc, out_channels=self.class_num, d1=8, d2=self.d2-2, z_dim=self.z_dim).to(self.device)
        elif self.is_HP and (not self.is_NS):
            self.models['F'] = CLF_L2Softmax(in_channels=self.nc, out_channels=self.class_num, d1=self.d1, d2=self.d2, z_dim=self.z_dim).to(self.device)
        else:     
            self.models['F'] = CLF_Softmax(in_channels=self.nc, out_channels=self.class_num, d1=self.d1, d2=self.d2, z_dim=self.z_dim).to(self.device)
        

        self.models['Q'] = SUNet(self.nc, self.nc).to(device)
        self.models['G'] = SUNetZ(self.nc, self.z_dim, self.nc).to(device)

        self.optims['F'] = torch.optim.Adam(
            self.models['F'].parameters(), lr=1e-3, betas=(0.9, 0.990))
        
        self.optims['Q'] = torch.optim.Adam(
            self.models['Q'].parameters(), lr=1e-3, betas=(0.9, 0.999))

        self.optims['G'] = torch.optim.Adam(
            self.models['G'].parameters(), lr=1e-3, betas=(0.9, 0.999))
        
        self.train_sets['train_y'] = RFdataset(device_ids=range(45), test_ids=[1,2,3,4], rand_max_SNR=self.train_snr, is_FIR=self.is_FIR)
        self.train_sets['train_c'] = RFdataset(device_ids=range(45), test_ids=[1,2,3,4], rand_max_SNR=self.train_snr, is_FIR=self.is_FIR)
        self.eval_sets['val'] = RFdataset(device_ids=range(45), test_ids=[5], rand_max_SNR=None)

        self.eval_sets['T1'] = RFdataset(device_ids=range(45, 54), test_ids=[1,2,3,4,5], SNR=None)
        self.eval_sets['T2'] = RFdataset(device_ids=range(45), test_ids=[6,7,8,9], SNR=None)
        self.eval_sets['T3'] = RFdataset(device_ids=range(45, 54), test_ids=[6,7,8,9], SNR=None)

        self.eval_sets['M1'] = RFdataset_MP(device_ids=range(5), test_ids=[1], rand_max_SNR=None)
        self.eval_sets['M2'] = RFdataset_MP(device_ids=range(5), test_ids=[1,2], rand_max_SNR=None)
        self.eval_sets['M3'] = RFdataset_MP(device_ids=range(5), test_ids=[1,2,3], rand_max_SNR=None)

        

        self.preprocessing()
        for key in self.dataloaders.keys():
            self.records[key] = {}
            self.records[key]['acc'] = 0.0
            self.records[key]['auc'] = 0.0
            self.records[key]['auc_list'] = []
            self.records[key]['eer_list'] = []
            self.records[key]['val_loss'] = []

        if not os.path.exists('./plots'):
            print('./plots', 'dose not exist')
            os.makedirs('./plots')


    def data_generator(self, data_key):
        while 1:
            for data in self.dataloaders[data_key]:
                yield data

    def signal2image(self, input):
        N, _, T, _ = input.shape
        input_img = input.view(N, 1, T, 2).permute(0, 3, 1, 2).flatten().view(N, -1, 16, 80)
        return input_img

    def train(self, epoch):
        self.logs = {}
        self.models['Q'].train()
        self.models['F'].train()
        self.models['G'].train()
        for i in range(self.max_iter):

            data_y = next(self.data_generator('train_y'))
            x1, y1 = data_y[self.data_idx], data_y[1]
            x1, y1 = x1.to(self.device), y1.to(self.device)
            N = len(x1)

            data_c = next(self.data_generator('train_c'))
            x2, y2 = data_c[self.data_idx], data_c[1]
            x2, y2 = x2.to(self.device), y2.to(self.device)

            L_Q = torch.zeros(1)
            L_G = torch.zeros(1)
            kld = torch.zeros(1)
            if self.lamda > 0:
                # Q/G step
                x1_bag = self.models['Q'](x1)
                scores1_bag = self.models['F'](x1_bag)
                scores1 = self.models['F'](x1, y1)
                logp1 = torch.log_softmax(scores1, dim=1)[range(N), y1]
                logp1_bag = torch.log_softmax(scores1_bag, dim=1)
                logp1_bag_y = logp1_bag[range(N), y1]
                py = torch.ones_like(logp1_bag_y) * 1/self.class_num
                logpy = py.log()
                k = self.epsilon
                kld = (logp1_bag_y-logpy).mean()
                kld[kld<k] = kld[kld<k]*0

                mse = F.mse_loss(x1_bag, x1)
                L_Q = self.alpha* kld + mse

                z1 = self.models['F'].features(x1)
                x1_rec = self.models['G'](x1_bag, z1)
                z1_rec = self.models['F'].features(x1_rec)
                if self.beta > 0.0:
                    L_G = F.mse_loss(x1_rec, x1) + self.beta * F.mse_loss(z1_rec, z1)
                else:
                    L_G = F.mse_loss(x1_rec, x1)
                L = L_Q + L_G
                self.optims['G'].zero_grad()
                self.optims['Q'].zero_grad()
                L.backward()
                self.optims['G'].step()
                self.optims['Q'].step()

            # F step
            
            if self.is_swap:
                # Background swap
                z1 = self.models['F'].features(x1).detach()
                x2_bag = self.models['Q'](x2)
                x12 = self.models['G'](x2_bag, z1)
            else:
                z1 = self.models['F'].features(x1).detach()
                x2_bag = self.models['Q'](x1)
                x12 = self.models['G'](x1_bag, z1).detach()

            scores1_rff = self.models['F'](x1, y1)
            scores2_rff = self.models['F'](x12, y1)

            L_rff = (1-self.lamda) * F.cross_entropy(scores1_rff, y1) + self.lamda * F.cross_entropy(scores2_rff, y2)
            self.optims['F'].zero_grad()
            L_rff.backward()
            self.optims['F'].step()


            if i % 100 == 0:
                self.logs['kld'] = kld.item()
                self.logs['L_Q'] = L_Q.item()
                self.logs['L_G'] = L_G.item()

                self.print_logs(epoch, i)
                if self.lamda > 0:
                    images = torch.cat([
                        self.signal2image(x1[:8])[:, 0:1, :, :],
                        self.signal2image(x1_bag[:8])[:, 0:1, :, :],
                        self.signal2image(x1_rec[:8])[:, 0:1, :, :],
                        self.signal2image((x1_rec[:8]-x1[:8])**2**0.5)[:, 0:1, :, :],
                        self.signal2image(x12[:8])[:, 0:1, :, :],
                        self.signal2image(x2_bag[:8])[:, 0:1, :, :],
                        self.signal2image(x2[:8])[:, 0:1, :, :],
                    ], dim=0)

                    tv.utils.save_image(images, './plots/vis.png', nrow=8)
                
        self.eval(epoch, eval_dataset='M3', mode=None, is_record=True)
        return 0.0
                

    def eval(self, epoch, eval_model = ['F'], eval_dataset = None, mode=None, is_record=True):
        self.logs = {}
        main_model = 'F'
        feature_dict = {}
        label_dict = {}
        distance_dict = {}
        correct_dict = {}
        if eval_dataset is None:
            eval_dataset = self.dataset
        
        for model in eval_model:
            self.models[model].eval()
            feature_dict[model] = []
            label_dict[model] = []
            distance_dict[model] = []
            correct_dict[model] = 0.0

        with torch.no_grad():
            for data in self.dataloaders[eval_dataset]:
                x, y = data[self.data_idx], data[1]
                x, y = x.to(self.device), y.to(self.device)
                N = len(x)
                for model in eval_model:
                    features = self.models[model].features(x)
                    feature_dict[model].append(features)
                    label_dict[model].append(y)
                if mode == 'clf':
                    scores = self.models[model].output(features)
                    pred_y = torch.argmax(scores, dim=1)
                    correct_dict[model] += torch.sum(pred_y == y).item()

        for model in eval_model:
            features = torch.cat(feature_dict[model], dim=0)
            labels = torch.cat(label_dict[model])
            intra_dist, inter_dist = inter_intra_dist(features.cpu().detach().numpy(), labels.cpu().numpy())
            distance_hist_plot(intra_dist, inter_dist, filename='./plots/{}_{}_dist_hist.png'.format(self.flag, eval_dataset))
            distance_dict[model] = [intra_dist, inter_dist]

        results = roc_plots(distance_dict, file_name='./plots/{}_{}_roc.png'.format(self.flag, eval_dataset))

        is_best = False
        auc, eer, _ = results[main_model]

        if mode == 'clf':
            acc = correct_dict[main_model] / len(self.datasets[eval_dataset])
            self.logs['acc'] = acc

        if auc >= self.records[eval_dataset]['auc']:
            if eval_dataset == 'val':
                is_best = True
            self.records[eval_dataset]['auc'] = auc
        
        self.logs['auc'] = auc
        self.logs['data'] = eval_dataset
        self.logs['eer'] = eer
        self.logs['best auc'] = self.records[eval_dataset]['auc']

        self.print_logs(epoch, 0)
        if is_record:
            self.records[eval_dataset]['auc_list'].append(auc)
            self.records[eval_dataset]['eer_list'].append(eer)

        return is_best 

if __name__ == '__main__':

    ## DR-RFF
    trainer = Trainer(train_snr=None,
                    device=0, d2=18, z_dim=512, 
                    lamda=0.5, alpha=10, beta=10, epsilon=0.0, is_NS=False, is_HP=True, is_FIR=False)

    ## ML-RFF
    # trainer = Trainer(train_snr=None,
    #                 device=0, d2=18, z_dim=512, 
    #                 lamda=0.0, alpha=10, beta=10, epsilon=0.0, is_NS=False, is_HP=True, is_FIR=False)

    ## AWGN
    # trainer = Trainer(train_snr=30,
    #                 device=0, d2=18, z_dim=512, 
    #                 lamda=0.0, alpha=10, beta=10, epsilon=0.0, is_NS=False, is_HP=True, is_FIR=False)

    trainer.run(load_best=True, retrain=False, is_del_loger=False)

    trainer.eval(0, eval_dataset='T1', mode=None, is_record=False)
    trainer.eval(0, eval_dataset='T2', mode=None, is_record=False)
    trainer.eval(0, eval_dataset='T3', mode=None, is_record=False)

    trainer.eval(0, eval_dataset='M1', mode=None, is_record=False)
    trainer.eval(0, eval_dataset='M2', mode=None, is_record=False)
    trainer.eval(0, eval_dataset='M3', mode=None, is_record=False)


                            