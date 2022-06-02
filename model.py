import torch 
import numpy as np
import timm
from utils import class_embedding, convert_index_to_word, read_glove_vecs, sentences_to_indices
import _pickle as pickle

class InceptionV4():
    def __init__(self) -> None:
        self.model = timm.create_model('inception_v4', pretrained=True)

class Embedder(torch.nn.Module):
    def __init__(self, basedir) -> None:
        super().__init__()
        self.basedir = basedir
        self.num_features = 1536
        self.final_linear = torch.nn.Linear(self.num_features, 1024)
        
    def embed_classes(self, classes_embed, batch_size):
        # Get top-k ids
        dataset_name = 'Aircraft'
        word2index, index2word, word2vec = read_glove_vecs(self.basedir + 'data/%s_glove6b_init_300d.npy' % dataset_name,
                                                            self.basedir + 'data/%s_dictionary.pkl' % dataset_name)
        word2index[index2word[0]] = len(word2index)
        classes = pickle.load(open(self.basedir + 'data/%s_classes.pkl' % dataset_name, 'rb'))
        classes = np.array(classes)
        indices = sentences_to_indices(classes, word2index, 3)
        indices = torch.tensor(indices)
        indices = torch.broadcast_to(indices, [batch_size, len(classes), 3])

        topk_cls = classes_embed
        topk_cls = torch.unsqueeze(topk_cls, 2)
        topk_cls = torch.broadcast_to(topk_cls, [topk_cls.shape[0], 5, 3])
        params = indices.permute(0,2,1)
        ids = topk_cls.permute(0,2,1)
        topk_cls = torch.gather(params, ids, batch_dims=-1)
        topk_cls = topk_cls.permute(0,2,1)

        add_emb = np.expand_dims(word2vec[0], 0)
        word2vec = np.append(word2vec, add_emb, axis=0)
        word2vec[0] = np.zeros((word2vec.shape[1]))
        emb = class_embedding(topk_cls, word2vec, word2index, 300)
        return emb

    def stack_features(self, layers_feature):
        pass

    def combine(self, layers_feature, classes_embed):
        layers_feature = self.stack_features(layers_feature)
        batch_size = layers_feature.shape[0]
        classes_embed = self.embed_classes(classes_embed, batch_size)
    
    def forward(self, layers_feature, classes_embed):
        x = self.combine(layers_feature, classes_embed)
        x = self.final_linear(x)
        return x
       

class Model(torch.nn.Module):
    def __init__(self, backbone, basedir):
        super().__init__()

        self.basedir = basedir
        # (N,3,299,299)->(N,1792,12,12)
        self.features = backbone
        
        # (N,1792,12,12)->(N,1792,1,1)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512, 100),
            torch.nn.LogSoftmax(dim=-1)
        )

        self.embed = Embedder(basedir)

    def forward(self, x):
        self.fmap = self.features.forward_features(x) # (N,3,300,300)->(N,1568,8,8)
        N = self.fmap.shape[0]
        for i in N:
            topk = torch.topk(self.features.forward_head(self.fmap)[i], k=10)
            topk_indices = topk.indices
        topk_classes = 0    

        x = self.embed(self.fmap, topk_classes)
        # x = self.avg_pool(self.fmap).reshape(N,-1) # (N,1920,9,9)->(N,1920,1,1)->(N,1920) 
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