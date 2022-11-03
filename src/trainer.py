import marveltoolbox as mt 
from .models import *
import torch
import torch.nn as nn
import torch.nn.functional as F
# import horovod.torch as hvd
from .dataset import *
from .evaluation import *

class RFFConfs(mt.BaseConfs):
    def __init__(self, train_snr, device=0, d1=None, d2=None, z_dim=32):
        self.train_snr = train_snr
        self.device = device
        self.device_ids = [device]
        self.d1 = d1
        self.d2 = d2
        self.z_dim = z_dim
        super().__init__()
    
    def get_dataset(self):
        self.dataset = 'zigbee'
        self.nc = 2
        self.batch_size = 256
        self.class_num = 54
        self.epochs = 200
        self.train_devices = range(45)
        self.train_ids = [1,2,3,4]

    def get_flag(self):
        self.eval_model = NS_CLF_Arcface_old
        self.data_idx = 0
        self.flag = 'Baseline-CNS-CNN-arcface-old-snr{}-nz{}'.format(self.train_snr. self.z_dim)

    def get_device(self):
        self.device = torch.device(
            "cuda:{}".format(self.device) if \
            torch.cuda.is_available() else "cpu")


class RFFTrainer(mt.BaseTrainer):
    def __init__(self):
        mt.BaseTrainer.__init__(self, self)
        if (not self.d1 is None) and (not self.d2 is None):
            self.models['C'] = self.eval_model(in_channels=self.nc, out_channels=self.class_num, d1=self.d1, d2=self.d2, z_dim=self.z_dim).to(self.device)
        else:
            self.models['C'] = self.eval_model(in_channels=self.nc, out_channels=self.class_num, z_dim=self.z_dim).to(self.device)

        self.optims['C'] = torch.optim.Adam(
            self.models['C'].parameters(), lr=1e-4, betas=(0.5, 0.99))
        # self.optims['C'] = torch.optim.Adam(
        #     self.models['C'].parameters(), lr=1e-3, betas=(0.9, 0.999))

        # self.datasets['train_MP'] = RFdataset_MP(device_ids=range(0,6), test_ids=[0,1,2,3], rand_max_SNR=None)
        # self.datasets['train'] = RFdataset(device_ids=range(45), test_ids=[5], rand_max_SNR=None)
        # self.datasets['open'] = RFdataset_MP(device_ids=range(7, 12), test_ids=[0,1,2,3], rand_max_SNR=None)

        self.eval_sets['MP0'] = RFdataset_MP(device_ids=range(6), test_ids=[0,1,2,3], rand_max_SNR=None)
        self.eval_sets['MP1'] = RFdataset_MP(device_ids=range(7,12), test_ids=[0,1,2,3], rand_max_SNR=None)
        self.eval_sets['MP-all'] = RFdataset_MP(device_ids=range(12), test_ids=[0,1,2,3], rand_max_SNR=None)
        self.train_sets['train'] = RFdataset(device_ids=range(45), test_ids=[1,2,3,4], rand_max_SNR=self.train_snr)
        self.eval_sets['val'] = RFdataset(device_ids=range(45), test_ids=[5], rand_max_SNR=None)
        self.eval_sets['open'] = RFdataset(device_ids=range(6), test_ids=[1,2,3,4,5], rand_max_SNR=None)

        # self.datasets['train'] = RFdataset(device_ids=self.train_devices, test_ids=self.train_ids, rand_max_SNR=self.train_snr)
        # self.datasets['close'] = RFdataset_MP(device_ids=range(0,12), test_ids=[0,1,2,3], rand_max_SNR=None)
        # self.datasets['close'] = RFdataset(device_ids=range(45), test_ids=[5], rand_max_SNR=None)

        self.preprocessing()

        self.records['acc'] = 0.0
        self.records['auc'] = 0.0

    def train(self, epoch):
        self.logs = {}
        self.models['C'].train()
        for i, data in enumerate(self.dataloaders['train']):
            # data_mp = next(iter(self.dataloaders['train']))

            x, y = data[self.data_idx], data[1]
            x, y = x.to(self.device), y.to(self.device)

            # x_nmp, y_nmp = data[self.data_idx], data[1]
            # x_nmp, y_nmp = x.to(self.device), y.to(self.device)
            scores = self.models['C'](x, y)
            loss_mp = F.cross_entropy(scores, y)

            # scores_nmp = self.models['C'](x_nmp, y_nmp)
            # loss_nmp = F.cross_entropy(scores_nmp, y_nmp)
            loss = loss_mp # + loss_nmp
            self.optims['C'].zero_grad()
            loss.backward()
            self.optims['C'].step()
            
            if i % 100 == 0:
                self.logs['Train Loss'] = loss.item()
                self.print_logs(epoch, i)

        return loss.item()
                
    def eval(self, epoch, eval_dataset = 'open'):
        self.logs = {}
        self.models['C'].eval()
        correct = 0.0
        test_loss = 0.0
        feature_list = []
        label_list = []

        with torch.no_grad():
            for data in self.dataloaders[eval_dataset]:
                x, y = data[self.data_idx], data[1]
                x, y = x.to(self.device), y.to(self.device)
                N = len(x)
                features = self.models['C'].features(x)
                feature_list.append(features)
                label_list.append(y)

                if eval_dataset == 'close':
                    scores = self.models['C'].output(features)
                    test_loss += F.cross_entropy(scores, y, reduction='sum').item()
                    pred_y = torch.argmax(scores, dim=1)
                    correct += torch.sum(pred_y == y).item()

        features = torch.cat(feature_list, dim=0)
        labels = torch.cat(label_list)
        intra_dist, inter_dist = inter_intra_dist(features.cpu().detach().numpy(), labels.cpu().numpy())
        distance_hist_plot(intra_dist, inter_dist, filename='./plots/{}_{}_dist_hist.png'.format(self.flag, eval_dataset))
        eer, roc_auc, thresh = get_auc_eer(intra_dist, inter_dist, plot_roc=True, filename='./plots/{}_{}_roc.png'.format(self.flag, eval_dataset))
        is_best = False

        if eval_dataset == 'close':
            acc = correct / len(self.datasets[eval_dataset])
            test_loss = test_loss/ len(self.datasets[eval_dataset])
                
            if acc >= self.records['acc']:
                is_best = True
                self.records['acc'] = acc
            self.logs['Test Loss'] = test_loss
            self.logs['acc'] = acc
            self.logs['auc'] = roc_auc
            self.logs['eer'] = eer
            self.logs['data'] = eval_dataset
            self.print_logs(epoch, 0)
        else:

            if roc_auc >= self.records['auc']:
                is_best = True
                self.records['auc'] = roc_auc
            self.logs['auc'] = roc_auc
            self.logs['eer'] = eer
            self.logs['data'] = eval_dataset
            self.print_logs(epoch, 0)

        return is_best
