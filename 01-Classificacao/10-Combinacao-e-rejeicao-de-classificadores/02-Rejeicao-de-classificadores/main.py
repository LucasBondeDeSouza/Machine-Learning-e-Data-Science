import pickle  # Biblioteca usada para salvar e carregar arquivos
import numpy as np  # Biblioteca para manipular listas de números (arrays)
from sklearn.svm import SVC  # Importa o modelo de Máquina de Vetores de Suporte (SVM)

# Carrega os dados pré-processados (informações dos clientes e respostas salvas em 'credit.pkl')
with open('credit.pkl', 'rb') as f:
    # Dados de entrada (características dos clientes) e saídas (se têm crédito aprovado ou não)
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Treina o modelo SVM com probabilidade ativada
svm = SVC(probability=True)  # Habilita a estimativa de probabilidades para prever a confiança
svm.fit(x_credit_treinamento, y_credit_treinamento)  # Treina o modelo com os dados de treinamento

# Salva o modelo SVM treinado em um arquivo
with open('svm_finalizado.sav', 'wb') as f:
    pickle.dump(svm, f)

# Junta os dados de treinamento e teste em um único conjunto para análises adicionais
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)  # Dados de entrada
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)  # Respostas esperadas

# Carrega os modelos de aprendizado de máquina previamente treinados
rede_neural = pickle.load(open('rede_neural_finalizado.sav', 'rb'))  # Modelo de Rede Neural
arvore = pickle.load(open('arvore_finalizado.sav', 'rb'))  # Modelo de Árvore de Decisão
svm = pickle.load(open('svm_finalizado.sav', 'rb'))  # Modelo SVM

# Seleciona um cliente da base de dados para prever se ele pagará o empréstimo
novo_registro = x_credit[1999]  # Seleciona o cliente de índice 1999

# Redimensiona o dado para o formato esperado pelos modelos (uma linha com várias colunas)
novo_registro = novo_registro.reshape(1, -1)  # Transforma em matriz de 1 linha e N colunas

# Faz previsões com cada modelo
resposta_rede_neural = rede_neural.predict(novo_registro)  # Resultado da Rede Neural
resposta_arvore = arvore.predict(novo_registro)  # Resultado da Árvore de Decisão
resposta_svm = svm.predict(novo_registro)  # Resultado do SVM

# Obtém as probabilidades (confiança) das previsões
probabilidade_rede_neural = rede_neural.predict_proba(novo_registro)  # Confiança da Rede Neural
confianca_rede_neural = probabilidade_rede_neural.max()  # Maior valor de confiança
print(confianca_rede_neural)

probabilidade_arvore = arvore.predict_proba(novo_registro)  # Confiança da Árvore de Decisão
confianca_arvore = probabilidade_arvore.max()  # Maior valor de confiança
print(confianca_arvore)

probabilidade_svm = svm.predict_proba(novo_registro)  # Confiança do SVM
confianca_svm = probabilidade_svm.max()  # Maior valor de confiança
print(confianca_svm)

# Inicializa contadores para decisões
paga = 0  # Contador de modelos que dizem que o cliente pagará
nao_paga = 0  # Contador de modelos que dizem que o cliente não pagará
confianca_minima = 0.99999  # Nível mínimo de confiança para considerar o modelo
algoritmos = 0  # Contador de algoritmos que atendem ao critério de confiança

# Verifica a previsão da Rede Neural com base na confiança
if confianca_rede_neural >= confianca_minima:
    algoritmos += 1  # Incrementa o número de modelos confiáveis
    if resposta_rede_neural[0] == 1:  # Verifica se a resposta é "não pagará"
        nao_paga += 1
    else:  # Caso contrário, considera que "pagará"
        paga += 1

# Verifica a previsão da Árvore de Decisão com base na confiança
if confianca_arvore >= confianca_minima:
    algoritmos += 1
    if resposta_arvore[0] == 1:
        nao_paga += 1
    else:
        paga += 1

# Verifica a previsão do SVM com base na confiança
if confianca_svm >= confianca_minima:
    algoritmos += 1
    if resposta_svm[0] == 1:
        nao_paga += 1
    else:
        paga += 1

# Decide o resultado final com base nos votos dos modelos confiáveis
if paga > nao_paga:  # Mais modelos dizem que o cliente pagará
    print(f'Cliente pagará o empréstimo, baseado em {algoritmos} algoritmos')
elif paga == nao_paga:  # Empate entre os modelos
    print(f'Empate, baseado em {algoritmos} algoritmos')
else:  # Mais modelos dizem que o cliente não pagará
    print(f'Cliente não pagará o empréstimo, baseado em {algoritmos} algoritmos')