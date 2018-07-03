#  Regressão de conjunto de dados com python

Algoritmo em Python para realizar regressão utilizando MLP sobre um conjunto de dados.

Para realizar a regressão do conjunto de dados, serão utilizadas três bibliotecas:

* [Pandas](https://pandas.pydata.org/): para ler o conjunto de dados;
* [Sklearn](http://scikit-learn.org/stable/index.html): para realizar a padronização dos dados e a regressão utilizando o modelo de [Redes Neurais](https://www.digitaltrends.com/cool-tech/what-is-an-artificial-neural-network/); e
* [Matplotlib](https://matplotlib.org/): para a geração de gráficos 2D a partir de uma matriz.

<hr>

## Construindo o algoritmo do zero

para isso será preciso primeiramente baixar o conjunto de dados [winequality-red.csv](./winequality-red.csv).
 
depois criar um arquivo denominado ```regressao.py``` e abrir em um editor de texto.
<hr>

### após abrir o arquivo, vamos começar a implementação do código propriamente dito: 

Importando as bibliotecas
````
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn import model_selection as ms
import matplotlib.pyplot as plt
````

Fazendo leitura dos dados
````
base = pd.read_csv('winequality-red.csv')
````

<hr>

### Pre-processando os atributos

Por meio da definição dos índices de cada tipo de atributo
````
numericos = [0,1,2,3,4,5,6,7,8,9,10]
````

indexando os índices dos atributos numericos à variável auxiliar
````
atrib_num = base.iloc[:, numericos].values
````

definindo o atributo classe, ou seja, a ```coluna 11``` do conjunto de dados
````
classe = base.iloc[:, 11].values
````

dividindo matrizes em subconjuntos aleatórios de treinamento e teste
````
splits = ms.train_test_split(atrib_num, classe, test_size=0.2)
X_train, X_test, y_train, y_test = splits
````

definindo a quantidade de camadas escondidas
````
sizes = [10, 20, 25, 30]
````

váriaveis para atribuir os resultados do treinamento e do teste e plotar no gráfico
````
trainR2 = []
testR2 = []
````

<hr>

### Validacao Estatistica dos resultados

Instanciando objeto para normalização Z-score (escore padrão (média e desvio padrão))
````
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
````

o método ```fit``` ajusta o modelo aos dados que está sendo fornecidos. É aí que o modelo "aprende" a partir dos dados.
````
scaler_previsores = scaler.fit(X_train)
````

o método ```transform``` realiza a padronização centralizando e dimensionando os dados os dados de treinamento e de teste
````
X_train = scaler_previsores.transform(X_train)
X_test = scaler_previsores.transform(X_test)
````

estrutura de repetição (Loop) para criar modelos Perceptron de regressão de multicamadas 
````
for size in sizes:
    model = MLPRegressor(activation="relu", solver='lbfgs',
                         hidden_layer_sizes=size,
                         max_iter=1000, shuffle=True, random_state=1)

    model.fit(X_train, y_train)

    expected = y_train
    predicted = model.predict(X_train)
    print ("------------------Hidden layer = " + str(size) + "-----------------")
    print ("Dados de treinamento")
    print ("MSE: " + str(metrics.mean_squared_error(expected, predicted)))
    print ("R2: " + str(metrics.r2_score(expected, predicted)))
    trainR2.append(metrics.r2_score(expected, predicted))

    expected = y_test
    predicted = model.predict(X_test)
    print ("Dados de teste")
    print ("MSE: " + str(metrics.mean_squared_error(expected, predicted)))
    print ("R2: " + str(metrics.r2_score(expected, predicted)))
    testR2.append(metrics.r2_score(expected, predicted))
````

plotando os resultados em um gráfico 2D 
````
plt.plot(sizes, trainR2, sizes, testR2)
plt.show()
````

<hr>

### FIM 
O código completo se encontra [aqui](./regressao.py)