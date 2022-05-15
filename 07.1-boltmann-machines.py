# importando as bibliotecas necessárias

import numpy as np                       # biblioteca para cálculos matemáticos
import pandas as pd                      # biblioteca para manipulação de dados
import torch                             # biblioteca para cálculos com GPUs
import torch.nn as nn                    # biblioteca para criação de redes neurais
import torch.nn.parallel                 # biblioteca para criação de redes neurais paralelas
import torch.optim as optim              # biblioteca para otimização de redes neurais
from torch.autograd import Variable      # biblioteca para criação de variáveis

# carregando os dados

# filmes
movies = pd.read_csv('./data/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
# usuários
users = pd.read_csv('./data/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
# avaliações dos usuários
ratings = pd.read_csv('./data/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# carregando os dados de treino e teste
train_set = pd.read_csv('./data/u1.base', delimiter='\t')
test_set = pd.read_csv('./data/u1.test', delimiter='\t')

# transformando os dados em um array para uso com o torch
train_set = np.array(train_set, dtype='int')
test_set = np.array(test_set, dtype='int')

# pegando o identificador máximo dos filmes e usuários
nb_users = int(max(max(train_set[:, 0]), max(test_set[:, 0])))

# pegando o identificador máximo das avaliações
nb_movies = int(max(max(train_set[:, 1]), max(test_set[:, 1])))

# função para conversão dos dados para o formato do torch
# será necessário lista de listas 

def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

# conversão dos dados
train_set = convert(train_set)
test_set = convert(test_set)

# conversão dos dados para o formato do torch tensor .float()
train_set = torch.FloatTensor(train_set)
test_set = torch.FloatTensor(test_set)

# conversão das avaliações para um formato binario
# de 1-5 em -1, 0, 1
train_set[train_set == 0] = -1
train_set[train_set == 1] =  0
train_set[train_set == 2] =  0
train_set[train_set >= 3] =  1

test_set[test_set == 0] = -1
test_set[test_set == 1] =  0
test_set[test_set == 2] =  0
test_set[test_set >= 3] =  1

# criando uma classe para a rede neural
class RBM():

    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

# parametros de treinamento
nv = len(train_set[0])
nh = 100
batch_size = 100
nb_ephochs = 10

# criação da classe RBM
rbm = RBM(nv, nh)

# treinamento da rede
for epoch in range(1, nb_ephochs + 1):
    
    train_loss = 0
    s = 0.
    
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = train_set[id_user:id_user+batch_size]
        v0 = train_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
    
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# teste do modelo
test_loss = 0
s = 0.

for id_user in range(nb_users):
    
    v = train_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
    
print('test loss: '+str(test_loss/s))