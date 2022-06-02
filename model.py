import torch 
import numpy as np
import timm

class InceptionV4():
    def __init__(self) -> None:
        self.model = timm.create_model('inception_v4', pretrained=True)

class Model(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()

        # (N,3,299,299)->(N,1792,12,12)
        self.features = backbone
        
        # (N,1792,12,12)->(N,1792,1,1)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifier = torch.nn.Sequential(
            ##############################################
            # WRITE YOUR CODE HERE
            # 여기서 self.features가 출력하는 모양을 flatten 했을 때
            # 노드 수를 계산하여 
            # 계산된 노드수->100으로 가는 완전 연결층을 구성하기
            torch.nn.Linear(backbone._fc.in_features, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512, 100),
            torch.nn.LogSoftmax(dim=-1)
            ##############################################
        )
    
    def forward(self, x):
        self.fmap = self.features.extract_features(x) # (N,3,300,300)->(N,1920,9,9)
        
        N = self.fmap.shape[0]
        x = self.avg_pool(self.fmap).reshape(N,-1) # (N,1920,9,9)->(N,1920,1,1)->(N,1920) 
        x = self.classifier(x) #(N,1920)->(N,100)

        return x

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2