import pandas as pd 
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# tentativa de melhorar o algoritmo
import numpy as np
from sklearn.model_selection import train_test_split

# seleciona os dados do arquivos
# data frame (df) - molde de dados
dados = pd.read_csv('CSV_Anfibios_Cortada.csv')

# a_trocar_estacao = {
#     'Primavera': 1,
#     'Inverno': 2,
#     'Verão': 3,
#     'Outono': 4
# }
# dados.estacao = dados.estacao.map(a_trocar_estacao)
# print(dados.estacao.value_counts())

# a_trocar_sentido = {
#     'RJ': 1,
#     'Inverno': 2,
#     'Verão': 3,
#     'Outono': 4
# }
# dados.Sentido = dados.Sentido.map(a_trocar_sentido)

# print(dados.value_counts())

Y_df = dados['mais_atropelado']  
X_df = dados[["Mês", "estacao", 
              "Trecho", "Sentido", "Trecho macro",
              "Tipo De Pistas", "Numero de Pistas",
              "Tipo de Pavimento", "Numero de Faixas",
              "Velocidade Maxima", "Tipo da Chuva","Intervencao",
              "Vazamento", "Agua?", "Vegetação Baixa?","capimAlto",
              "Arbustos","Floresta","Local","Com filhotes?",
              "Faixa Horário da Coleta","Faixa de Quilometragem"]]

# Para saber quantos sao sim (1) e quantos sao nao (0)
# temos 12915 - 0 e 181 - 1
valores_y = Y_df.value_counts()

# variavel categoricas - dividimos em diferentes categorias : dummies
# O Naive bayes só aceita valores numéricos, entao aqui transformamos tudo para que sejam numericos 
Xdummies_df = pd.get_dummies(X_df)
# print(Xdummies_df)

# retorna um array com os valores 
x = Xdummies_df.values
y = Y_df.values 

# divide os dados 2/3 treino, 1/3 teste  e 10% pra validacao
porcentagem_treino = 0.8
porcentagem_teste  = 0.1

tamanho_treino    = int(porcentagem_treino * len(y))
tamanho_teste     = int(porcentagem_teste * len(y))

tamanho_validacao = len(y) - tamanho_treino - tamanho_teste

# 0 ate 80% -> treino
treino_x = x[0:tamanho_treino]
treino_y = y[0:tamanho_treino]

# 80 ate 90% - teste
fim_do_teste = tamanho_treino + tamanho_teste
teste_x      = x[tamanho_treino:fim_do_teste]
teste_y      = y[tamanho_treino:fim_do_teste]

# 90 ate 100% - validacao
validacao_x = x[fim_do_teste:]
validacao_y = y[fim_do_teste:]

# Definindo o modelo de machine learning
def fit_predict_model(nome, modelo):
    # treina modelo
    modelo.fit(treino_x, treino_y)
    
    # testa modelo
    teste = modelo.predict(teste_x) # Pega as previsoes do modelo para os valores X de teste
    acertos = teste == teste_y

    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_x)

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    msg = "A taxa de acerto do {} foi: {:.2f}".format(nome, taxa_de_acerto)
    print(msg)
    
    cm = confusion_matrix(teste_y, teste, labels=None, sample_weight=None, normalize=None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    print(disp.plot())

    return taxa_de_acerto


modelo_multinomial = MultinomialNB ()
resultado_multinomial = fit_predict_model("Multinomial", modelo_multinomial)

modelo_gaussiano = GaussianNB ()
resultado_gaussiano = fit_predict_model("gaussiano", modelo_gaussiano)

modelo_bernoulli = BernoulliNB ()
resultado_bernoulli = fit_predict_model("bernoulli", modelo_bernoulli)


# Algoritmo bobo
# algoritimo de base chuta o que é mais frequente
acerto_base = max(Counter(validacao_y).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_y)
print("Taxa de acerto base: {:.2f}".format(taxa_de_acerto_base))

# separando treino e teste de forma aleatoria
SEED = 5
np.random.seed(SEED)
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.35,
                                                         stratify = y)

print("redividindo os dados")

modelo_multinomial = MultinomialNB ()
resultado_multinomial = fit_predict_model("Multinomial", modelo_multinomial)

modelo_gaussiano = GaussianNB ()
resultado_gaussiano = fit_predict_model("gaussiano", modelo_gaussiano)

modelo_bernoulli = BernoulliNB ()
resultado_bernoulli = fit_predict_model("bernoulli", modelo_bernoulli)
