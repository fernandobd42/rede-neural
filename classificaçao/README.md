#  Classificação de conjunto de dados com python

Algoritmo em Python utilizando Redes neurais para classificar uma base de dados que contém atributos: Numéricos e nominais. O programa realiza a validação dos resultados utilizando k-fold. Como o algoritmo MLP é baseado em pesos aleatórios o experimento foi realizado várias vezes, ao fim foi calculado média e desvio padrão da taxa de sucesso para todas as execuções.

Para realizar a classificação do conjunto de dados (do inglês dataset), será utilizado três bibliotecas:

* [NumPy](http://www.numpy.org/): para gerar o conjunto de dados concatenando atributos, e no fim calcular a média e o desvio padrão das taxas de sucesso do K-fold;
* [Pandas](https://pandas.pydata.org/): para ler o conjunto de dados; e
* [Sklearn](http://scikit-learn.org/stable/index.html): para realizar a padronização dos dados e a classifição utilizando o modelo de [Redes Neurais](https://www.digitaltrends.com/cool-tech/what-is-an-artificial-neural-network/).

<hr>

## Construindo o algoritmo do zero

para isso vai ser preciso primeiro baixar o conjunto de dados [flags.csv](./flags.csv).
 
depois crie um arquivo denominado ```classificador.py``` e abra em um editor de texto.
<hr>

## após abrir o arquivo, vamos começar a implementação do código propriamente dito: 

Importando as bibliotecas
````
import numpy as np
import pandas as pd
import copy
````

#### Definir o modelo de aprendizagem que será utilizado

No nosso caso será utilizado o modelo <b>Redes Neurais</b>
````
modelo_desejado = 'Redes Neurais'
#modelo_desejado = 'Arvores de Decisão'
#modelo_desejado = 'KNN'
#modelo_desejado = 'Naive Bayes'

if modelo_desejado == 'Redes Neurais':
    from sklearn.neural_network import MLPClassifier
    classificador = MLPClassifier(max_iter = 1000, tol = 0.0001)
elif modelo_desejado == 'Arvores de Decisão':
    from sklearn.tree import DecisionTreeClassifier
    classificador = DecisionTreeClassifier(criterion = 'entropy')
elif modelo_desejado == 'KNN':
    from sklearn.neighbors import KNeighborsClassifier
    classificador = KNeighborsClassifier(n_neighbors=3)
elif modelo_desejado == 'Naive Bayes':
    from sklearn.naive_bayes import GaussianNB
    classificador = GaussianNB()
````

Fazendo leitura dos dados
````
base = pd.read_csv('flags.csv')
````

Ordenando os dados pela classe
````
base = base.sort_values(by=base.keys()[-1])
````

<hr>

#### Pre-processando os atributos
 
Por meio da definição dos índices de cada tipo de atributo
````
numericos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26]
nominais = [0, 18, 28, 29]
````

indexando os índices dos atributos numericos e nominais à variáveis auxiliar
````
atrib_num = base.iloc[:, numericos].values
atrib_nom = base.iloc[:, nominais].values
````

definindo o atributo classe, ou seja, a coluna (27) do conjunto de dados, a qual contém o atributo utilizado para a classificação
````
classe = base.iloc[:, 27].values
````

convertendo os atributos nominais para numericos
````
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

classe = labelencoder.fit_transform(classe)

for i in range(len(nominais)):
      atrib_nom[:,i] = labelencoder.fit_transform(atrib_nom[:,i])

atrib_nom = atrib_nom.astype(int)
````

convertendo os atributos nominais que agora sao inteiros em binarios
````
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()

atrib_nom = onehotencoder.fit_transform(atrib_nom).toarray()
````

concatenando os atributos e gerando o conjunto de dados pradronizado
````
previsores = np.concatenate((atrib_num, atrib_nom), axis = 1)
````

<hr>

### Validacao Estatistica dos resultados

Instanciando objeto para normalização Z-score (escore padrão (média e desvio padrão))
````
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn.metrics import confusion_matrix
````

Descobrindo o número de instâncias da base
````
n_inst = base.shape[0]
````

Implementando k-fold (validação cruzada), ordenando os dados pela classe
````
n_folds = 2
kfold = []
for i in range(n_inst):
      kfold.append(i % n_folds)  
````

Estrutura de repetição (Loop) para validação cruzada (K-fold)
````
Rates = []
for pasta in range(n_folds):
      # A cada iteração deve-se copiar os previsores novamente
      # Pois a normalização z-score altera os dados
      previsoresValidacao = copy.deepcopy(previsores)
      
      # Descobre quais instâncias estão na pasta de treino
      # E quais instâncias estão na pasta de teste
      train = []
      test = []
      for i in range(n_inst):
            if kfold[i] == pasta:
                  test.append(i)
            else:
                  train.append(i)
      
      # A cada iteração a z-score normaliza TODOS os dados, para isso ela obtém a
      # média e o desvio padrão dos dados de TREINAMENTO (MUITO IMPORTANTE)
      scaler_previsores = scaler.fit(previsoresValidacao[train])
      previsoresValidacao = scaler_previsores.transform(previsoresValidacao)
      
      # Treinamento do classificador com as instâncias de treino
      classificador.fit(previsoresValidacao[train], classe[train])
      # Predição dos rótulos das instâncias de teste
      classe_predita = classificador.predict(previsoresValidacao[test])
      # Compara os rótulos preditos com os rótulos reais
      MC = confusion_matrix(classe[test], classe_predita)
      Rate = MC.diagonal().sum() / MC.sum()
      print(MC)
      print(Rate)
      Rates.append(100*Rate)
````

Calculando a média e o desvio padrão das taxas de sucesso do K-fold
````
media = np.mean(Rates)
desvio = np.std(Rates)
print('media = %.2f %%, Desvio = %.2f %%' %(media,desvio))
````

### FIM 
O código completo se encontra [aqui](./classificador.py)