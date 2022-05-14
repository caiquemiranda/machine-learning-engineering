'''

SOM ( self organizing map ) algoritmos
algoritmo de agrupamento // não supervisionado

'''

# importando bibliotecas necessárias

import pandas as pd                                       # biblioteca para manipulação de dados
import numpy as np                                        # biblioteca para cálculos matemáticos
from sklearn.preprocessing import MinMaxScaler            # biblioteca para normalização dos dados
from minisom import MiniSom                               # biblioteca para criação do SOM
from pylab import bone, pcolor, colorbar, plot, show      # biblioteca para plotar gráficos

# importação da base de dados
dataset = pd.read_csv('./data/Credit_Card_Applications.csv')

# separando os dados em atributos e classes
X = dataset.iloc[ :, :-1].values
y = dataset.iloc[ :, -1].values

# escalonamento dos dados entre 0 e 1 (normalização)
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

'''
criação do objeto SOM
x e y são as dimensões da matriz do mapa
input_len é o número de atributos do dataset
sigma é o tamanho do mapa (raio, a partir do nó central)
learning_rate é a taxa de aprendizado (quantos pesos são atualizados por iteração)
'''
# definindo os parâmetros do SOM
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)

# criação do objeto SOM
som.random_weights_init(X)

# treinamento do modelo
som.train_random(data=X, num_iteration=100)

# criação do mapa
bone()
pcolor(som.distance_map().T)
colorbar()

# marcação dos registros no mapa
markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)

# visualização do mapa
show();

# identificação dos fraudes
mappings = som.win_map(X)

# revertendo a trasnformação de escalonamento para os valores originais
fraudes = np.concatenate((mappings[(8,1)], mappings[(9,2)]), axis=0)
fraudes = sc.inverse_transform(fraudes)

# visualização dos registros que são fraudes
fraudes