# importando as bibliotecas necessárias
import numpy as np                                  # biblioteca para cálculos matemáticos
import pandas as pd                                 # biblioteca para manipulação de dados
import matplotlib.pyplot as plt                     # biblioteca para plotagem de gráficos
from sklearn.preprocessing import MinMaxScaler      # biblioteca para normalização de dados
from tensorflow.keras.models import Sequential      # biblioteca para construção de redes neurais
from tensorflow.keras.layers import Dense           # biblioteca para camadas densas
from tensorflow.keras.layers import LSTM            # biblioteca para camadas LSTM
from tensorflow.keras.layers import Dropout         # biblioteca para camadas Dropout

# importando os dados de treino e teste
train = pd.read_csv('./data/Google_Stock_Price_Train.csv')
test = pd.read_csv('./data/Google_Stock_Price_Test.csv')

# transformando os dados de treino em arrays
train_set = train.iloc[:, 1:2].values

# normalização entre 0 e 1
scale = MinMaxScaler(feature_range=(0, 1))
train_set_scaled = scale.fit_transform(train_set)
train_set_scaled

# processo de treino 
X_train = []   # intervalo de tempo 60 dias
y_train = []   # previsão do proximo dia

for i in range(60, 1258):
    X_train.append(train_set_scaled[i-60: i, 0])
    y_train.append(train_set_scaled[i, 0])

# conversão de listas para arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# transformando os arrays em matrizes para o Keras
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# construindo a rede neural LSTM com Keras

# inicializando a rede
regressor = Sequential()

# primeira camada LSTM
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))

# dropout para evitar overfitting
regressor.add(Dropout(0.2))

# segunda camada LSTM
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# terceira camada LSTM
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# quarta camada LSTM
regressor.add(LSTM(units=50,))
regressor.add(Dropout(0.2))

# camada de saída
regressor.add(Dense(units=1))
# compilando a rede
regressor.compile(optimizer='adam', loss='mean_squared_error')

# treinando a rede
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# previsões do modelo treinado

# transformando os dados de teste
test_set = test.iloc[:, 1:2].values

# concatenando os dados de treino e teste e um unico dataset
dataset_total = pd.concat((train['Open'], test['Open']), axis=0)

# preparando os dados para a previsão [60 dias antes], e em array
inputs = dataset_total[len(dataset_total) - len(test) - 60:].values

# ajuste do input para o formato esperado pelo Keras
inputs = inputs.reshape(-1, 1)

# normalização dos dados
inputs = scale.transform(inputs)

# preparando os dados para a previsão [60 dias antes]
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60: i, 0])

X_test = np.array(X_test)

# transformando os dados para o formato esperado pelo Keras
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# previsão do modelo treinado
previ_stock_price = regressor.predict(X_test)

# revertendo a normalização
previ_stock_price = scale.inverse_transform(previ_stock_price)

# comparação dos previsões com os dados de teste
# avaliando modelo
plt.plot(test_set, color='red', label='Dados reais Google')
plt.plot(previ_stock_price, color='blue', label='Previsão Google')
plt.title('Previsão do preço do Google')
plt.xlabel('Tempo')
plt.ylabel('Preço')
plt.legend()
plt.show()