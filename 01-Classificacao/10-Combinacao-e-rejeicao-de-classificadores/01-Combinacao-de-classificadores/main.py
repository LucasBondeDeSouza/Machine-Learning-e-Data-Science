import pickle  # Biblioteca usada para salvar e carregar arquivos
import numpy as np  # Biblioteca para manipular listas de números (arrays)

# Carrega os dados pré-processados (informações de clientes e respostas salvas em 'credit.pkl')
with open('credit.pkl', 'rb') as f:
    # Dados de entrada (características dos clientes) e saídas (se têm crédito aprovado ou não)
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Junta os dados de treinamento e teste em um único conjunto
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)  # Dados dos clientes
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)  # Respostas (sim ou não)

# Carrega os modelos de aprendizado de máquina já treinados
rede_neural = pickle.load(open('rede_neural_finalizado.sav', 'rb'))  # Modelo de Rede Neural
arvore = pickle.load(open('arvore_finalizado.sav', 'rb'))  # Modelo de Árvore de Decisão
svm = pickle.load(open('svm_finalizado.sav', 'rb'))  # Modelo SVM (Máquina de Vetores de Suporte)

# Escolhe um cliente para prever se pagará o empréstimo
novo_registro = x_credit[1999]  # Seleciona o cliente de índice 1999

# Redimensiona o dado para o formato esperado pelos modelos (uma linha com várias colunas)
novo_registro = novo_registro.reshape(1, -1)

# Faz previsões com cada modelo
resposta_rede_neural = rede_neural.predict(novo_registro)  # Resposta da Rede Neural
resposta_arvore = arvore.predict(novo_registro)  # Resposta da Árvore de Decisão
resposta_svm = svm.predict(novo_registro)  # Resposta do SVM

# Conta quantas vezes os modelos preveem "pagará" ou "não pagará"
paga = 0  # Contador para "pagará"
nao_paga = 0  # Contador para "não pagará"

# Verifica a resposta de cada modelo e atualiza os contadores
if resposta_rede_neural[0] == 1:  # Se a rede neural prevê "não pagará"
    nao_paga += 1
else:  # Caso contrário, prevê "pagará"
    paga += 1

if resposta_arvore[0] == 1:  # Se a árvore de decisão prevê "não pagará"
    nao_paga += 1
else:  # Caso contrário, prevê "pagará"
    paga += 1

if resposta_svm[0] == 1:  # Se o SVM prevê "não pagará"
    nao_paga += 1
else:  # Caso contrário, prevê "pagará"
    paga += 1

# Decide o resultado final baseado na maioria das previsões
if paga > nao_paga:  # Maioria prevê que o cliente pagará
    print('Cliente pagará o empréstimo')
elif paga == nao_paga:  # Empate entre as previsões
    print('Empate')
else:  # Maioria prevê que o cliente não pagará
    print('Cliente não pagará o empréstimo')