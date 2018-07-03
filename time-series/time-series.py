# Importando as bibliotecas
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Fazendo leitura dos dados
base = pd.read_csv("BTC.csv")


#%% Pre-processando os atributos

# identificando e apagando a linha quando a coluna 5 for nula
nulos = pd.isnull(base.iloc[:,5])
base.drop(base[nulos].index, inplace = True)

# filtrando somente os valores da coluna 5 a atribuindo à variável Y
Y = base.pop(base.keys()[5])

# filtrando somente os valores da coluna 0 a atribuindo à variável datas
datas = base.pop(base.keys()[0])

# convertendo as datas do tipo string para o tipo date
datas = pd.to_datetime(datas, format="%Y/%m/%d")

# criando uma matriz y_data através da lista Y
y_data = np.array(Y)

# convertendo as datas em números ordinais e atribuindo à variável x_data
X_data = datas.map(dt.datetime.toordinal)

# criando uma matriz x_data através da lista x_data de números ordinais
X_data = np.array(X_data).reshape(len(X_data), 1)

# definindo o tamanho a janela de intervalo
window = 10

# definindo o comprimento da matriz y_data
end = y_data.shape[0]

# váriaveis para atribuir o série temporal verdadeira e a prevista para plotar no gráfico
y_pred = []
y_true = []


#%% Validacao Estatistica dos resultados

# estrutura de repetição (Loop) para criar as séries temporais verdadeira e prevista
for i in range(1, end-window):

    print ("Iteracao = " + str(i))
    X_train = X_data[i: i+window]
    y_train = y_data[i: i+window]

    X_test = X_data[i+window]
    y_test = y_data[i+window]

    model = MLPRegressor(activation="tanh", solver='lbfgs',
                         hidden_layer_sizes=(100, 50), max_iter=100,
                         shuffle=True, random_state=1)
    model.fit(X_train, y_train)

    y_pred.append(model.predict([X_test]))
    y_true.append(y_test)

# criando a matriz y_pred através da lista y_pred que contem os dados previstos
y_pred = np.array(y_pred)

# criando a matriz y_true através da lista y_true que contem os dados verdadeiros
y_true = np.array(y_true)

# plotando os resultados em um gráfico 2D 
plt.plot(range(y_pred.shape[0]), y_pred)
plt.plot(range(y_true.shape[0]), y_true)

plt.show()
