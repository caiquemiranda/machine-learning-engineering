# importação das bibliotecas necessarias

import pandas as pd                                             # biblioteca para manipulação de dados
import numpy as np                                              # biblioteca para cálculo matemático
import seaborn as sns                                           # biblioteca para visualização de dados
from sklearn.neural_network import MLPClassifier                # biblioteca para criação de modelo de rede neural
from sklearn.model_selection import train_test_split            # biblioteca para divisão de dados em conjuntos de treinamento e teste
from sklearn.preprocessing import LabelEncoder, OneHotEncoder   # biblioteca para codificação de dados
from sklearn.compose import make_column_transformer             # biblioteca para codificação de dados
from sklearn.preprocessing import StandardScaler                # biblioteca para normalização dos dados
from tensorflow.keras.models import Sequential                  # biblioteca para criação de modelo sequencial
from tensorflow.keras.layers import Dense                       # biblioteca para criação de camadas densas
from sklearn.metrics import confusion_matrix                    # biblioteca para cálculo da matriz de confusão

# carregamento dos dados
dataset = pd.read_csv('./data/Churn_Modelling.csv')

# pre processamento dos dados
# separação dos atributos que serão utilizados para treinamento e atributos que serão utilizados para predição
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# transformação dos dados categóricos em dados numéricos
labelencoder_X1 = LabelEncoder()
X[ :, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[ :, 2] = labelencoder_X2.fit_transform(X[:, 2])

# transformação dummy variables para os dados categóricos
onehotencoder = make_column_transformer((OneHotEncoder(categories='auto', sparse='False'), [1]), remainder='passthrough')
X = onehotencoder.fit_transform(X)

# remoção de atributos que foram transformados em dummy variables
X = X[:, 1:]

# divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# entrada com 11 atributos => camada oculta com 6 neurônios => camada de saída com 1 neurônio
# função de ativação oculta(relu)oculta(relu)saida(sigmoide)
# número de épocas = 100
# otimização = adam (gradiente descendente stocastico) achar o melhor valor para otimização dos pesos
# loss = binary_crossentropy
# metrica = acurácia


# criação do modelo de rede neural
nn_model = Sequential()

# definindo parametros da rede neural
nn_model.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
nn_model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
nn_model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# compilando o modelo
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# treinamento do modelo
nn_model.fit(X_train, y_train, batch_size=10, epochs=100)

# previsão dos dados de teste
y_pred = nn_model.predict(X_test)

# transformação dos dados de teste em dados binários
y_pred = (y_pred > 0.5)

# matriz de confusão
cmm = confusion_matrix(y_test, y_pred)


# visualização da matriz de confusão com seaborn
sns.heatmap(cmm, annot=True, cmap='Blues', fmt='g');