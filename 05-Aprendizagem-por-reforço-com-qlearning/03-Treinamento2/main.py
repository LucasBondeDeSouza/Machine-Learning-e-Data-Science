import gym  # Biblioteca para criar e testar ambientes de aprendizado por reforço
import random  # Gera valores aleatórios para a exploração de ações
from IPython.display import clear_output  # Limpa a saída no Jupyter Notebook
import numpy as np  # Trabalha com operações matemáticas e matrizes

# Cria o ambiente do jogo "Taxi-v3", onde um táxi deve pegar e deixar passageiros corretamente
env = gym.make('Taxi-v3', render_mode="ansi")

# Reinicia o ambiente para começar o jogo do estado inicial
env.reset()

# Exibe a representação visual do ambiente no terminal
print(env.render())

# Cria a Q-table (tabela que armazena o valor aprendido para cada estado e ação)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hiperparâmetros do algoritmo Q-learning
alpha = 0.1  # Taxa de aprendizado (o quanto o agente aprende a cada passo)
gamma = 0.6  # Fator de desconto (importância das recompensas futuras)
epsilon = 0.1  # Probabilidade de escolher ações aleatórias (exploração)

# Executa 100.000 episódios de treinamento
for i in range(100000):
    estado, _ = env.reset()  # Reinicia o ambiente e obtém o estado inicial
    penalidades, recompensas = 0, 0  # Inicializa contadores
    done = False  # Variável que indica se o episódio terminou

    while not done:
        # Escolhe uma ação: explora aleatoriamente ou segue a melhor estratégia aprendida
        if random.uniform(0, 1) < epsilon:
            acao = env.action_space.sample()  # Ação aleatória (exploração)
        else:
            acao = np.argmax(q_table[estado])  # Melhor ação conhecida (exploração guiada)

        # Executa a ação escolhida e recebe a resposta do ambiente
        proximo_estado, recompensa, terminated, truncated, info = env.step(acao)
        done = terminated or truncated  # Verifica se o episódio terminou

        # Atualiza a Q-table usando a fórmula do Q-learning
        q_antigo = q_table[estado, acao]  # Valor antigo da ação naquele estado
        proximo_maximo = np.max(q_table[proximo_estado])  # Melhor valor do próximo estado

        # Fórmula do Q-learning para atualizar o valor da ação tomada
        q_novo = (1 - alpha) * q_antigo + alpha * (recompensa + gamma * proximo_maximo)
        q_table[estado, acao] = q_novo  # Armazena o novo valor

        # Conta penalidades (movimentos errados)
        if recompensa == -10:
            penalidades += 1

        estado = proximo_estado  # Atualiza o estado atual

    # Exibe o progresso a cada 100 episódios
    if i % 100 == 0:
        clear_output(wait=True)
        print('Episódio: ', i)

print('Treinamento Concluído')  # Mensagem final indicando que o treinamento acabou