# importando as bibliotecas necessárias

import numpy as np                         # biblioteca para cálculos matemáticos
import pandas as pd                        # biblioteca para manipulação de dados 
import torch                               # biblioteca para cálculos com GPUs
import torch.nn as nn                      # biblioteca para criação de redes neurais 
import torch.nn.parallel                   # biblioteca para criação de redes neurais paralelas
import torch.optim as optim                # biblioteca para otimização de redes neurais
from torch.autograd import Variable        # biblioteca para criação de variáveis
import torch.utils.data                    # biblioteca para criação de dados

# carregando os dados
# filmes
movies = pd.read_csv('./data/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')

# usuários
users = pd.read_csv('./data/users.dat', sep='::', header=None, engine='python', encoding='latin-1')

# avaliações dos usuários
ratings = pd.read_csv('./data/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# carregando os dados de treino e teste
training_set = pd.read_csv('./data/u1.base', sep='\t')
test_set = pd.read_csv('./data/u1.test', sep='\t')

# transformando os dados em um array numpy
training_set = np.array(training_set, dtype='int')
test_set = np.array(test_set, dtype='int')

# pegando o numero máximo de usuários e filmes
nb_users = int(max(max(training_set[ :, 0]), max(test_set[ :, 0])))
nb_movies = int(max(max(training_set[ :, 1]), max(test_set[ :, 1])))

# função para transformação dos dados de treino e teste para uma lista(users) de listas(movies)
def convert(data):
    new_data = []
    
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    
    return new_data

# transformando os dados de treino e teste para uma lista(users) de listas(movies)
training_set = convert(training_set)
test_set = convert(test_set)

# convertendo os dados de treino e teste para o formato torch
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
class SAE(nn.Module):

    def __init__(self, ):
        super(SAE, self).__init__()           # chamando o construtor da classe mãe
        
        # definindo parametros da camada de entrada
        self.fc1 = nn.Linear(nb_movies, 20)    # 20, numeros de features da camada de entrada
        self.fc2 = nn.Linear(20, 10)           # 10, numeros de features da camada oculta
        self.fc3 = nn.Linear(10, 20)           # 20, numeros de features da camada de saída
        self.fc4 = nn.Linear(20, nb_movies)    # nb_movies, numeros de features da camada de saída
        self.activation = nn.Sigmoid()         # função de ativação

    def forward(self, x):
        
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        
        return x

# instanciando a classe SAE
sae = SAE()

# parametros da rede
criteon = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)
nb_epochs = 200

# treinando a rede
for epoch in range(1, nb_epochs +1):
    
    train_loss = 0
    s = 0.

    for id_user in range(nb_users):
        
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()

        if torch.sum(target.data > 0) > 0:
        
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criteon(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            optimizer.step()

    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))


# testando o modelo
test_loss = 0
s = 0.
for id_users in range(nb_users):
    
    input = Variable(training_set[id_users]).unsqueeze(0)
    target = Variable(test_set[id_users]).unsqueeze(0)

    if torch.sum(target.data > 0) > 0:
        
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criteon(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.

print('test loss: ' + str(test_loss/s))
