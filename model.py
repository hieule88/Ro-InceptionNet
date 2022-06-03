import torch 
import numpy as np
import timm
from utils import class_embedding, convert_index_to_word, read_glove_vecs, sentences_to_indices
import _pickle as pickle
import tensorflow as tf
from timm.models.inception_v4 import BasicConv2d
class InceptionV4():
    def __init__(self) -> None:
        self.model = timm.create_model('inception_v4', pretrained=False, num_classes= 100)

class Embedder(torch.nn.Module):
    def __init__(self, basedir, classes) -> None:
        super().__init__()
        self.basedir = basedir
        self.num_features = 1536
        self.final_linear = torch.nn.Linear(self.num_features, 1024)
        self.classes = classes
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.avg_pool_final = torch.nn.AdaptiveAvgPool2d(output_size=(1536,1))
        self.cnn0 = BasicConv2d(384, 1536, kernel_size=1, stride=1)
        self.cnn1 = BasicConv2d(1024, 1536, kernel_size=1, stride=1)
        self.linear1 = torch.nn.Linear(10, 1)
        self.linear2 = torch.nn.Linear(2, 1)
        self.relu = torch.nn.ReLU()
    def embed_classes(self, logits):
        # Get top-k ids
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_name = 'Aircraft'
        word2index, index2word, word2vec = read_glove_vecs(self.basedir + '/data/%s_glove6b_init_300d.npy' % dataset_name,
                                                            self.basedir + '/data/%s_dictionary.pkl' % dataset_name)
        word2index[index2word[0]] = len(word2index)
        classes = pickle.load(open(self.basedir + '/data/%s_classes.pkl' % dataset_name, 'rb'))
        classes = np.array(self.classes)
        indices = sentences_to_indices(classes, word2index, 3)
        indices = torch.tensor(indices, device=device)
        indices = torch.broadcast_to(indices, [logits.shape[0], len(classes), 3])
        topk_cls = torch.topk(logits, k=10)[1]
        topk_cls = torch.unsqueeze(topk_cls, 2)
        topk_cls = torch.broadcast_to(topk_cls, [topk_cls.shape[0], 10, 3])
        params = indices.permute(0,2,1)
        ids = topk_cls.permute(0,2,1)

        topk_cls = torch.gather(params, -1, ids)
        topk_cls = topk_cls.permute(0,2,1)
        add_emb = np.expand_dims(word2vec[0], 0)
        word2vec = np.append(word2vec, add_emb, axis=0)
        word2vec[0] = np.zeros((word2vec.shape[1]))
        word2vec = torch.tensor(word2vec, device=device)
        emb = class_embedding(topk_cls, word2vec, 300)
        return emb

    def stack_features(self, layers_feature):
        alpha = 0.1
        features, mixed7a, mixed6a, to_mixed5a = layers_feature
        
        to_mixed5a = self.cnn0(to_mixed5a)
        mixed6a_stack = self.cnn1(mixed6a)

        mixed6a_stack = self.avg_pool(mixed6a_stack).reshape(features.shape[0],-1)
        to_mixed5a = self.avg_pool(to_mixed5a).reshape(features.shape[0],-1)
        mixed7a = self.avg_pool(mixed7a).reshape(features.shape[0],-1)
        features = self.avg_pool(features).reshape(features.shape[0],-1)
        layers_feature = torch.stack((to_mixed5a, mixed6a_stack, mixed7a, features), dim=2)
        # layers_feature = torch.stack((to_mixed5a.mul(alpha), mixed6a_stack.mul(alpha), mixed7a.mul(alpha), features.mul(1-3*alpha)), dim=2)
        layers_feature = self.avg_pool_final(layers_feature).reshape(features.shape[0],-1)
        return layers_feature

    def combine(self, layers_feature, logits):
        
        topk = self.embed_classes(logits)
        topk = topk.permute(0,2,1)
        topk = self.linear1(topk)
        topk = self.relu(topk).reshape(topk.shape[0],-1)
        layers_feature = self.stack_features(layers_feature)
        combine_feature = torch.stack((topk, layers_feature), dim=2)
        combine_feature = self.linear2(combine_feature)
        combine_feature = self.relu(combine_feature).reshape(topk.shape[0],-1)
        return combine_feature
    def forward(self, layers_feature, logits):
        x = self.combine(layers_feature, logits)
        x = self.final_linear(x)
        return x
       

class Model(torch.nn.Module):
    def __init__(self, backbone, basedir, classes):
        super().__init__()

        self.classes = classes
        
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

        self.embed = Embedder(basedir, classes)

    def forward(self, x):
        self.fmap, mixed7a, mixed6a, to_mixed5a = self.features.forward_features(x) # (N,3,300,300)->(N,1568,8,8)
        N = self.fmap.shape[0]
        logits = self.features.forward_head(self.fmap)
        
        x = self.embed([self.fmap, mixed7a, mixed6a, to_mixed5a], logits)
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