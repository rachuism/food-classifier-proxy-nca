import logging, imp
import dataset
import utils
import proxynca
import net
from PIL import Image
import PIL.Image

import torch
import numpy as np
import matplotlib
matplotlib.use('agg', warn=False, force=True)
import matplotlib.pyplot as plt
import time
import argparse
from torch.autograd import Variable
import evaluation


'''De momento tomo que no hace falta
dl_ev = torch.utils.data.DataLoader(
    dataset.Birds(
        root = 'predict_data', 
        labels = list(range(args.nb_classes, 2 * args.nb_classes)),
        is_extracted = args.cub_is_extracted,
        transform = dataset.utils.make_transform(is_train = False)
    ),
    batch_size = args.sz_batch,
    shuffle = False,
    num_workers = args.nb_workers,
    pin_memory = True
)
'''
'''
pic = Image.open("Laysan_Albatross_0016_64734298.jpg")
pic = dataset.utils.make_transform(pic)
#pix = np.array(pic.getdata())

#value = torch.from_numpy(pix)
test_value = pic.cuda()
test_value = test_value.float()

model = net.bn_inception(pretrained = True)
#net.embed(model, sz_embedding=args.sz_embedding) #Iniciara el modelo?
model = model.cuda()

prediction = model(test_value)

print(pix)'''
'''
def predict_batchwise(model, dataloader):
    with torch.no_grad():
        X, Y = zip(*[
            [x, y] for X, Y in dataloader
                for x, y in zip(
                    model(X.cuda()).cpu(), 
                    Y
                )
        ])
    return torch.stack(X), torch.stack(Y)

image = Image.open("Scissor_tailed_Flycatcher_0002_2820701856.jpg")
transform = dataset.utils.make_transform()
image = transform(image)
image = image.unsqueeze(0).float()
    
image = Variable(image)
model = net.bn_inception()
net.embed(model, 64)
model.load_state_dict(torch.load('saved/saved_mode.pth'))
#net.embed(model, 100)
model = model.cuda()
image = image.cuda()
out = model.forward(image)
results = torch.exp(out).data.topk(5)

classes = np.array(results[1][0], dtype=np.int)
probs = Variable(results[0][0]).data
'''

def predict_batchwise(model, dataloader):
    with torch.no_grad():
        X, Y = zip(*[
            [x, y] for X, Y in dataloader
                for x, y in zip(
                    model(X.cuda()).cpu(), 
                    Y
                )
        ])
    return torch.stack(X), torch.stack(Y)

model = net.bn_inception()
net.embed(model, 64)
model.load_state_dict(torch.load('saved/saved_mode.pth'))
#net.embed(model, 100)
model = model.cuda()

dl_ev = torch.utils.data.DataLoader(
    dataset.Birds(
        root = 'upmc20', 
        labels = list(range(0, 20)),
        transform = dataset.utils.make_transform(is_train = False)
    ),
    batch_size = 64,
    shuffle = False,
    num_workers = 16,
    pin_memory = True
)

print(dl_ev)

X, T = predict_batchwise(model, dl_ev)

print(X[1200])
print(T[1200])

Y = evaluation.assign_by_euclidian_at_k(X, T, 8)

print(Y[1200])

#################################
'''
import argparse

parser = argparse.ArgumentParser(description='Training inception V2' + 
    ' (BNInception) on CUB200 with Proxy-NCA loss as described in '+ 
    '`No Fuss Distance Metric Learning using Proxies.`'
)
# export directory, training and val datasets, test datasets
parser.add_argument('--cub-root', 
    default='cub200',
    help = 'Path to root CUB folder, containing the images folder.'
)
parser.add_argument('--cub-is-extracted', action = 'store_true',
    default = False,
    help = 'If `images.tgz` was already extracted, do not extract it again.' +
        ' Otherwise use extracted data.'
)
parser.add_argument('--embedding-size', default = 64, type = int,
    dest = 'sz_embedding',
    help = 'Size of embedding that is appended to InceptionV2.'
)
parser.add_argument('--number-classes', default = 100, type = int,
    dest = 'nb_classes',
    help = 'Number of first [0, N] classes used for training and ' + 
        'next [N, N * 2] classes used for evaluating with max(N) = 100.'
)
parser.add_argument('--batch-size', default = 32, type = int,
    dest = 'sz_batch',
    help = 'Number of samples per batch.'
)
parser.add_argument('--lr-embedding', default = 1e-5, type = float,
    help = 'Learning rate for embedding.'
)
parser.add_argument('--lr-inception', default = 1e-3, type = float,
    help = 'Learning rate for Inception, excluding embedding layer.'
)
parser.add_argument('--lr-proxynca', default = 1e-3, type = float,
    help = 'Learning rate for proxies of Proxy NCA.'
)
parser.add_argument('--weight-decay', default = 5e-4, type = float,
    dest = 'weight_decay',
    help = 'Weight decay for Inception, embedding layer and Proxy NCA.'
)
parser.add_argument('--epsilon', default = 1e-2, type = float,
    help = 'Epsilon (optimizer) for Inception, embedding layer and Proxy NCA.'
)
parser.add_argument('--gamma', default = 1e-1, type = float,
    help = 'Gamma for multi-step learning-rate-scheduler.'
)
parser.add_argument('--epochs', default = 20, type = int,
    dest = 'nb_epochs',
    help = 'Number of training epochs.'
)
parser.add_argument('--log-filename', default = 'example',
    help = 'Name of log file.'
)
parser.add_argument('--gpu-id', default = 0, type = int,
    help = 'ID of GPU that is used for training.'
)
parser.add_argument('--workers', default = 16, type = int,
    dest = 'nb_workers',
    help = 'Number of workers for dataloader.'
)

args = parser.parse_args()

dl_ev = torch.utils.data.DataLoader(
    dataset.Birds(
        root = args.cub_root, 
        labels = list(range(args.nb_classes, 2 * args.nb_classes)),
        is_extracted = args.cub_is_extracted,
        transform = dataset.utils.make_transform(is_train = False)
    ),
    batch_size = args.sz_batch,
    shuffle = False,
    num_workers = args.nb_workers,
    pin_memory = True
)
dataloader = dl_ev
def predict_batchwise(model, dataloader):
    with torch.no_grad():
        X, Y = zip(*[
            [x, y] for X, Y in dataloader
                for x, y in zip(
                    model(X.cuda()).cpu(), 
                    Y
                )
        ])
    return torch.stack(X), torch.stack(Y)
model.eval()

# calculate embeddings with model, also get labels (non-batch-wise)
X, T = predict_batchwise(model, dataloader)

# calculate NMI with kmeans clustering
nmi = evaluation.calc_normalized_mutual_information(
    T, 
    evaluation.cluster_by_kmeans(
        X, 100
    )
)
logging.info("NMI: {:.3f}".format(nmi * 100))

# get predictions by assigning nearest 8 neighbors with euclidian
Y = evaluation.assign_by_euclidian_at_k(X, T, 8)

print(X[0])
print(Y[0])

'''
