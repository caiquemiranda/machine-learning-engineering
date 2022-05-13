# importação das bibliotecas necessárias
import numpy as np                                              # biblioteca para cálculo matemático
import pandas as pd                                             # biblioteca para manipulação de dados
import matplotlib.pyplot as plt                                 # biblioteca para plotagem de gráficos
from sklearn.impute import SimpleImputer                        # biblioteca para preenchimento de valores faltantes
from sklearn.preprocessing import LabelEncoder, OneHotEncoder   # bibliotecas para codificação de variáveis categóricas
from sklearn.compose import ColumnTransformer                   # biblioteca para transformação de variáveis categóricas
from sklearn.model_selection import train_test_split            # biblioteca para divisão de dados em conjuntos de treinamento e teste
from sklearn.preprocessing import StandardScaler                # biblioteca para normalização dos dados

# importação do arquivo CSV
dataset = pd.read_csv('./data/Data.csv')  

# variaveis independentes
X = dataset.iloc[ : ,: -1].values                     # .values para transformar em array

# variavel dependente (y)
y = dataset.iloc[ :, -1].values

''' 
tratamento de dados
preenchimento de valores faltantes com o metodo do SimpleImputer
substituição dos valores faltantes pela média, valores em X das colunas 1 e 2
'''
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[ :, 1:3])  
X[ :, 1:3] = imputer.transform(X[ :, 1:3])

# enconding de variaveis categoricas
labelEncoder_X = LabelEncoder()
X[ :, 0] = labelEncoder_X.fit_transform(X[ :, 0])

'''
codificação de variaveis categoricas (dummy variables) - OneHotEncoder
transformação para que valor das variaveis sejam equivalentes
'''
ct = ColumnTransformer([('Country', OneHotEncoder(),[0])], remainder='passthrough')
X = ct.fit_transform(X)

# transformação de variaveis dependentes (y), para numéricas
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# divisão de conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

'''
feature scaling
escalonamento dos dados para que possam ser comparados com outros dados
feature scaling é feita para que os dados estejam na mesma escala
'''
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)