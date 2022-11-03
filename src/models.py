import torch
import torch.nn as nn
import torch.nn.functional as F
is_debug = False
if not is_debug:
    from .ZigBee_processing import * 
    from .ArcFace import *
else:
    from ZigBee_processing import * 
    from ArcFace import *

class NormalizedModel(nn.Module):
    def __init__(self) -> None:
        super(NormalizedModel, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # mean = input.mean(dim=2, keepdim=True).repeat(1, 1, 1280, 1)
        # std = input.std(dim=2, keepdim=True).repeat(1, 1, 1280, 1)
        mean = input.mean()
        std = input.std()
        normalized_input = (input - mean)/std
        return normalized_input

class BaseCLF(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, d=64):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (1x1280x2)
            NormalizedModel(),
            nn.Conv2d(in_channels=in_channels, out_channels=2*d, kernel_size=(10, 1), stride=1, padding=(5, 0)),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((4,1)),

            # State (128x320x2)
            nn.Conv2d(in_channels=2*d, out_channels=4*d, kernel_size=(3,2), stride=1, padding=1),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((4,1)),

            # State (256x80x3)
            nn.Conv2d(in_channels=4*d, out_channels=4*d, kernel_size=(80, 3), stride=1, padding=0),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5)
        )
            # outptut of main module --> State (1024x4x4)

        self.output = nn.Linear(4*d, out_channels)

        
    def forward(self, input):
        out = self.features(input)
        out = self.output(out)
        return out
    
    def features(self, input):
        N = len(input)
        out = self.main_module(input).view(N, -1)
        return out

class BaseCLF2(nn.Module):
    def __init__(self, in_channels=2, out_dim=1, d=4):
        super().__init__()
        self.d = d
        self.main_module = nn.Sequential(
            # Image (2x16x80)
            # NormalizedModel(),
            nn.Conv2d(in_channels=in_channels, out_channels=self.d, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d),
            nn.LeakyReLU(0.2),

            # State (dx16x80)
            nn.Conv2d(in_channels=self.d, out_channels=self.d*2, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d*2),
            nn.LeakyReLU(0.2),

            # State (2dx16x80)
            nn.Conv2d(in_channels=self.d*2, out_channels=self.d*4, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d*4),
            nn.LeakyReLU(0.2),

            # State (4dx8x40)
            nn.Conv2d(in_channels=self.d*4, out_channels=self.d*8, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d*8),
            nn.LeakyReLU(0.2),

            # State (8dx8x40)
            nn.Conv2d(in_channels=self.d*8, out_channels=self.d*16, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d*16),
            nn.LeakyReLU(0.2),

            # State (16dx4x20)
            nn.Conv2d(in_channels=self.d*16, out_channels=self.d*32, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d*32),
            nn.LeakyReLU(0.2),

            # State (32dx2x10)
            nn.Conv2d(in_channels=self.d*32, out_channels=out_dim, kernel_size=(2, 10), stride=1, padding=(0, 0)),
            # nn.BatchNorm2d(out_dim),

            
        )
            # outptut of main module --> State (1024x4x4)
        # self.output = nn.Linear(z_dim, out_channels)

    def forward(self, input, labels=None):
        out = self.features(input)
        # out = self.output(out)
        return out
    
    def features(self, input):
        N, _, T, _ = input.shape
        input_img = input.view(N, 1, T, 2).permute(0, 3, 1, 2).flatten().view(N, -1, 16, 80)
        out = self.main_module(input_img).view(N, -1)
        return out

class BaseCLF3(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, d=4):
        super().__init__()
        self.d = d
        self.main_module = nn.Sequential(
            # Image (2x16x80)
            # NormalizedModel(),
            nn.Conv2d(in_channels=in_channels, out_channels=self.d, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d),
            nn.LeakyReLU(0.2),

            # State (dx16x80)
            nn.Conv2d(in_channels=self.d, out_channels=self.d*2, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d*2),
            nn.LeakyReLU(0.2),

            # State (2dx16x80)
            nn.Conv2d(in_channels=self.d*2, out_channels=self.d*4, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d*4),
            nn.LeakyReLU(0.2),

            # State (4dx8x40)
            nn.Conv2d(in_channels=self.d*4, out_channels=self.d*8, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d*8),
            nn.LeakyReLU(0.2),

            # State (8dx8x40)
            nn.Conv2d(in_channels=self.d*8, out_channels=self.d*16, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d*16),
            nn.LeakyReLU(0.2),

            # State (16dx4x20)
            nn.Conv2d(in_channels=self.d*16, out_channels=self.d*32, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            # nn.BatchNorm2d(self.d*32),
            nn.LeakyReLU(0.2),

            # State (32dx2x10)
            nn.Conv2d(in_channels=self.d*32, out_channels=512, kernel_size=(2, 10), stride=1, padding=(0, 0)),
            # nn.BatchNorm2d(512),
            nn.Dropout(p=0.5)
        )
            # outptut of main module --> State (1024x4x4)
        self.output = nn.Linear(512, out_channels)

        
    def forward(self, input, labels=None):
        out = self.features(input)
        out = self.output(out)
        return out
    
    def features(self, input):
        N, _, T, _ = input.shape
        input_img = input.view(N, 1, T, 2).permute(0, 3, 1, 2).flatten().view(N, -1, 16, 80)
        out = self.main_module(input_img).view(N, -1)
        return out

class Freq_processing(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_offset_estimation = BaseCLF(1, out_channels=1)
    
    def forward(self, input):
        N, _, T, _ = input.shape
        freq = self.freq_offset_estimation(input)
        out = freq_compensation(input.view(N, T, -1), freq.view(-1))
        return out, freq

class Phase_processing(nn.Module):
    def __init__(self):
        super().__init__()
        self.phase_offset_estimation = BaseCLF(1, out_channels=1)
    
    def forward(self, input):
        N, _, T, _ = input.shape
        phase = self.phase_offset_estimation(input)
        out = phase_compensation(input.view(N, T, -1), phase.view(-1))
        return out, phase

class Synchronization(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.freq_estimation = BaseCLF2(2, out_dim=1, d=d)
        self.phase_estimation = BaseCLF2(2, out_dim=1, d=d)
    
    def forward(self, input):
        N, _, T, _ = input.shape
        freq_offset = self.freq_estimation(input.view(N, 1, T, 2)).view(-1)
        out = freq_compensation(input.view(N, T, -1), freq_offset)
        phase_offset = self.phase_estimation(out.view(N, 1, T, 2)).view(-1)
        out = phase_compensation(out.view(N, T, -1), phase_offset)
        return out.view(N, 1, T, 2), freq_offset, phase_offset

class SynchronizationVis(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.freq_estimation = BaseCLF(1, out_channels=1,d=d)
        self.phase_estimation = BaseCLF(1, out_channels=1,d=d)
    
    def forward(self, input):
        N, _, T, _ = input.shape
        freq_offset = self.freq_estimation(input.view(N, 1, T, 2)).view(-1)
        out = freq_compensation(input.view(N, T, -1), freq_offset)
        phase_offset = self.phase_estimation(out.view(N, 1, T, 2)).view(-1)
        out = phase_compensation(out.view(N, T, -1), phase_offset)
        return out.view(N, 1, T, 2), freq_offset, phase_offset

###########################
class CLF_yjb(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (1x1280x2)
            NormalizedModel(),
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(10, 1), stride=1, padding=(5, 0)),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((4,1)),

            # State (128x320x2)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,2), stride=1, padding=1),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((4,1)),

            # State (256x80x3)
            nn.Conv2d(in_channels=256, out_channels=z_dim, kernel_size=(80, 3), stride=1, padding=0),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5)
        )
            # outptut of main module --> State (1024x4x4)

        self.output = nn.Linear(z_dim, out_channels)

        
    def forward(self, input, labels=None):
        out = self.features(input)
        out = self.output(out)
        return out
    
    def features(self, input):
        N = len(input)
        out = self.main_module(input).view(N, -1)
        return out

class NS_CLF_Arcface_old(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.freq_processing = Freq_processing()
        self.phase_processing = Phase_processing()

        self.d = 32
        self.main_module = nn.Sequential(
            # Image (2x16x80)
            # NormalizedModel(),
            nn.Conv2d(in_channels=in_channels, out_channels=self.d, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d),
            nn.LeakyReLU(0.2),

            # State (dx16x80)
            nn.Conv2d(in_channels=self.d, out_channels=self.d*2, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d*2),
            nn.LeakyReLU(0.2),

            # State (2dx16x80)
            nn.Conv2d(in_channels=self.d*2, out_channels=self.d*4, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d*4),
            nn.LeakyReLU(0.2),

            # State (4dx8x40)
            nn.Conv2d(in_channels=self.d*4, out_channels=self.d*8, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d*8),
            nn.LeakyReLU(0.2),

            # State (8dx8x40)
            nn.Conv2d(in_channels=self.d*8, out_channels=self.d*16, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d*16),
            nn.LeakyReLU(0.2),

            # State (16dx4x20)
            nn.Conv2d(in_channels=self.d*16, out_channels=self.d*32, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d*32),
            nn.LeakyReLU(0.2),

            # State (32dx2x10)
            nn.Conv2d(in_channels=self.d*32, out_channels=z_dim, kernel_size=(2, 10), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(z_dim),

            
        )
            # outptut of main module --> State (1024x4x4)

        self.output = ArcMarginProduct(z_dim, out_channels, s=10, m=0.5)


    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out, labels)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment1, freq1 = self.freq_processing(input.view(N, 1, T, 2))
        segment2, phase = self.phase_processing(segment1.view(N, 1, T, 2))
        nn_input = segment2.view(N, 1, T, 2).permute(0, 3, 1, 2).flatten().view(N, -1, 16, 80)
        out = self.main_module(nn_input).view(N, -1)
        return out

class NS_CLF_Softmax_old(nn.Module):
    def __init__(self, in_channels=2, out_channels=10, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.freq_processing = Freq_processing()
        self.phase_processing = Phase_processing()

        self.d = 32
        self.main_module = nn.Sequential(
            # Image (2x16x80)
            # NormalizedModel(),
            nn.Conv2d(in_channels=in_channels, out_channels=self.d, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d),
            nn.LeakyReLU(0.2),

            # State (dx16x80)
            nn.Conv2d(in_channels=self.d, out_channels=self.d*2, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d*2),
            nn.LeakyReLU(0.2),

            # State (2dx16x80)
            nn.Conv2d(in_channels=self.d*2, out_channels=self.d*4, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d*4),
            nn.LeakyReLU(0.2),

            # State (4dx8x40)
            nn.Conv2d(in_channels=self.d*4, out_channels=self.d*8, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d*8),
            nn.LeakyReLU(0.2),

            # State (8dx8x40)
            nn.Conv2d(in_channels=self.d*8, out_channels=self.d*16, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d*16),
            nn.LeakyReLU(0.2),

            # State (16dx4x20)
            nn.Conv2d(in_channels=self.d*16, out_channels=self.d*32, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d*32),
            nn.LeakyReLU(0.2),

            # State (32dx2x10)
            nn.Conv2d(in_channels=self.d*32, out_channels=z_dim, kernel_size=(2, 10), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(z_dim),

            
        )
            # outptut of main module --> State (1024x4x4)

        self.output = nn.Linear(z_dim, out_channels)


    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment1, freq1 = self.freq_processing(input.view(N, 1, T, 2))
        segment2, phase = self.phase_processing(segment1.view(N, 1, T, 2))
        nn_input = segment2.view(N, 1, T, 2).permute(0, 3, 1, 2).flatten().view(N, -1, 16, 80)
        out = self.main_module(nn_input).view(N, -1)
        return out

class CLF_Softmax(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, d1=8, d2=32, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = nn.Linear(z_dim, out_channels)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment = input
        out = self.main_module.features(segment).view(N, -1)
        return out

class CLF_L2Softmax(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = ArcMarginProduct(z_dim, out_channels, s=10, m=0.0)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out, labels)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment = input
        out = self.main_module.features(segment).view(N, -1)
        return out

class CLF_Arcface(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = ArcMarginProduct(z_dim, out_channels, s=10, m=0.5)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out, labels)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment = input
        out = self.main_module.features(segment).view(N, -1)
        return out

class NS_CLF_Softmax(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.synchronization = Synchronization(d=d1)
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = nn.Linear(z_dim, out_channels)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment, _, _ = self.synchronization(input)
        out = self.main_module.features(segment).view(N, -1)
        return out

class NS_CLF_Softmax_Vis(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.synchronization = SynchronizationVis(d1)
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = nn.Linear(z_dim, out_channels)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment, _, _ = self.synchronization(input)
        out = self.main_module.features(segment).view(N, -1)
        return out

class NS_CLF_L2Softmax(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.synchronization = Synchronization(d=d1)
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = ArcMarginProduct(z_dim, out_channels, s=10, m=0.0)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out, labels)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment, _, _ = self.synchronization(input)
        out = self.main_module.features(segment).view(N, -1)
        return out

class NS_CLF_L2Softmax_Vis(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.synchronization = SynchronizationVis()
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = ArcMarginProduct(z_dim, out_channels, s=10, m=0.0)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out, labels)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment, _, _ = self.synchronization(input)
        out = self.main_module.features(segment).view(N, -1)
        return out

class NS_CLF_Arcface(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.synchronization = Synchronization(d=d1)

        self.d = 16
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = ArcMarginProduct(z_dim, out_channels, s=10, m=0.5)
        # self.output = nn.Linear(512, out_channels)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out, labels)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment, _, _ = self.synchronization(input)
        out = self.main_module.features(segment).view(N, -1)
        return out







if __name__ == '__main__':
    x = torch.randn(10, 1, 1280,2)
    test_model = NS_CLF_Softmax(in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512)
    print(test_model(x).shape)
    test_model = NS_CLF_L2Softmax(in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512)
    print(test_model(x).shape)
    test_model = CLF_L2Softmax(in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512)
    print(test_model(x).shape)
    test_model = CLF_Softmax(in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512)
    print(test_model(x).shape)

