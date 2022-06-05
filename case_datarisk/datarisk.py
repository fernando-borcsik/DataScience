import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# Abrir o DataFrame de Treino
df = pd.read_csv('treino.csv')

# Não embuti as plotagens aqui no código, pois minha análise foi feita no jupyter notebook,
# com o comando %matplotlib inline. Com o heatmap abaixo, mapeei as entradas nulas, para ter uma noção
# da quantidade de informações faltantes.
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# Eliminar linhas com valores nulos
dfna = df.dropna()

# Eliminar linhas com util_linhas_inseguras > 1 (Impossível ser maior que 100%). As exclusões de linhas não
# comprometem seriamente a análise, visto que representam uma porcentagem pequena dos dados totais. Se
# representassem uma grande parte dos dados, seria interessante eliminar as colunas, ou usar o próprio
# machine learning para completá-las.
drop_util = dfna[dfna['util_linhas_inseguras']>1].index
dfclean = dfna.drop(drop_util)

#Separação para treino
X = dfclean.drop('inadimplente', axis=1)
y = dfclean['inadimplente']

# Aqui optei por utilizar uma random forest. No jupyter notebook testei KNN, trees e SVM com GridSearch,
# porém comparando a confiabilidade das classificações com classification report e confusion matrix,
# RandomForest apresentou a maior confiabilidade, já a partir de 600 árvores. Fiz com 1000, já que
# o tempo de máquina é bem menor nesse método do que no SVM com GridSearch.
rfc = RandomForestClassifier(n_estimators=1000, verbose=3)
rfc.fit(X, y)

# Abrindo o DataFrame com dados de teste
test = pd.read_csv('teste.csv')

# A limpeza dos dados de teste deve ser melhor pensada, em relação aos dados de treino. Não faz sentido
# eliminar linhas ou colunas, simplesmente. Nenhum caso pode ser ignorado. Assim, busquei encontrar relações
# entre as grandezas do dataframe, sem muito sucesso. É possível verificar, por exemplo, no heatmap abaixo que não há
# forte correlação entre as colunas dos dados faltantes e os demais.
sns.heatmap(test.corr())

# De fato, tal heatmap, seja nos dados de treino ou teste, apontam a necessidade de diferentes abordagens. Uma
# simples regressão numérica seria falha nesse contexto.
# Assim, o critério para dados faltantes foi simplesmente substituir pela média. No caso de util_linhas_inseguras > 1,
# utilizei a média dos valores menores ou iguais a 1.
def imput_sal(sal):
    mean_sal = test['salario_mensal'].mean()
    if pd.isnull(sal):
        return mean_sal
    else:
        return sal

def imput_dep(dep):
    mean_dep = np.rint(test['numero_de_dependentes'].mean())
    if pd.isnull(dep):
        return mean_dep
    else:
        return dep

def imput_util(util):
    mean_util = test[test['util_linhas_inseguras']<=1]['util_linhas_inseguras'].mean()
    if util > 1:
        return mean_util
    else:
        return util

# Criação de uma cópia de test, chamada out, onde as predições serão acrescentadas, sem alteração dos dados faltantes
out = test
test['salario_mensal'] = test['salario_mensal'].apply(imput_sal)
test['numero_de_dependentes'] = test['numero_de_dependentes'].apply(imput_dep)
test['util_linhas_inseguras'] = test['util_linhas_inseguras'].apply(imput_util)

# Predições e criação da coluna no DataFrame test
pred = rfc.predict(test)
out['inadimplente'] = pred

# Salvando o csv resultante
out.to_csv('out.csv')


