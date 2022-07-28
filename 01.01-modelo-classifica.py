# importação das bibliotecas necessárias

import numpy as np                                              # biblioteca para cálculo matemático
import pandas as pd                                             # biblioteca para manipulação de dados
import matplotlib.pyplot as plt                                 # biblioteca para plotar gráficos
import seaborn as sns                                           # biblioteca para visualização de dados
from sklearn.impute import SimpleImputer                        # biblioteca para preenchimento de valores faltantes
from sklearn.preprocessing import LabelEncoder, OneHotEncoder   # bibliotecas para codificação de variáveis categóricas
from sklearn.compose import ColumnTransformer                   # biblioteca para transformação de variáveis categóricas
from sklearn.model_selection import train_test_split            # biblioteca para divisão de dados em conjuntos de treinamento e teste
from sklearn.preprocessing import StandardScaler                # biblioteca para escalonamento dos dados
from sklearn.linear_model import LogisticRegression             # biblioteca para criação de modelo de regressão logística
from sklearn.metrics import confusion_matrix                    # biblioteca para cálculo da matriz de confusão
from matplotlib.colors import ListedColormap                    # biblioteca para plotar cores específicas

# carregamento dos dados
dataset = pd.read_csv('./data/Social_Network_Ads.csv')

# separação dos atributos de entrada (X) e saída (y)
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# separação dos dados em conjunto de treinamento (75%) e teste (25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# feature scaling das variáveis de entrada
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# criação do modelo de classificação
modelo_rl = LogisticRegression(random_state=0)
modelo_rl.fit(X_train, y_train)

# previsão dos resultados
y_predict = modelo_rl.predict(X_test)

# comparação dos resultados através da matriz de confusão
matriz_confusao = confusion_matrix(y_test, y_predict)
matriz_confusao

# visualização da matriz de confusão com seaborn
sns.heatmap(matriz_confusao, annot=True, cmap='Blues', fmt='g');
plt.title('Matriz de Confusão');
plt.xlabel('Valor previsto');
plt.ylabel('Valor real');
plt.show();


# visualização dos resultados graficamente para os dados de treinamento
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

plt.contourf(X1, X2, modelo_rl.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Classificador (treinamento)')
plt.xlabel('Idade')
plt.ylabel('Salário')
plt.legend()
plt.show()

# visualização dos resultados graficamente para os dados de teste
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

plt.contourf(X1, X2, modelo_rl.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Classificador (treinamento)')
plt.xlabel('Idade')
plt.ylabel('Salário')
plt.legend()
plt.show()
