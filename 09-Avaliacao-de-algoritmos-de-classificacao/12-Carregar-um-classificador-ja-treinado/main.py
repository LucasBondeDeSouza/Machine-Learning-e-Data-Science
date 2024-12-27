import pickle  # Biblioteca usada para salvar e carregar arquivos
import numpy as np  # Biblioteca usada para manipulação de arrays (listas de números)

# Carrega os dados que foram pré-processados e salvos em um arquivo 'credit.pkl'
with open('../credit.pkl', 'rb') as f:
    # x_credit_treinamento e x_credit_teste são os dados de entrada (características dos clientes)
    # y_credit_treinamento e y_credit_teste são as respostas (se o cliente tem ou não crédito aprovado)
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Junta os dados de treinamento e teste em um único conjunto de dados para análise.
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)  # Junta as características (dados de entrada)
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)  # Junta as respostas (o que esperamos que o modelo aprenda)

# Carrega os modelos que já foram treinados e salvos
rede_neural = pickle.load(open('rede_neural_finalizado.sav', 'rb'))  # Carrega o modelo de rede neural
arvore = pickle.load(open('arvore_finalizado.sav', 'rb'))  # Carrega o modelo de árvore de decisão
svm = pickle.load(open('svm_finalizado.sav', 'rb'))  # Carrega o modelo de SVM (Máquinas de Vetores de Suporte)

# Seleciona um novo registro (cliente) da base de dados para prever o crédito aprovado
novo_registro = x_credit[1999]  # Seleciona o cliente de índice 1999 na lista de dados

# Ajusta a forma do dado para que ele tenha o formato necessário para a previsão
novo_registro = novo_registro.reshape(1, -1)  # Redimensiona para um único cliente (1 linha, várias colunas)

# Imprime os dados do novo cliente que será analisado
print(novo_registro)

# Usa os modelos treinados para prever se o cliente vai ter ou não o crédito aprovado
# A previsão retorna uma resposta (exemplo: 0 ou 1, onde 1 significa que o crédito é aprovado)
print(rede_neural.predict(novo_registro))  # Previsão com o modelo de rede neural
print(arvore.predict(novo_registro))  # Previsão com o modelo de árvore de decisão
print(svm.predict(novo_registro))  # Previsão com o modelo de SVM