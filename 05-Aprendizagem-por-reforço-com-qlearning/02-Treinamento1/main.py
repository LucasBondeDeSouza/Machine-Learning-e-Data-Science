import gym  # Importa a biblioteca Gym, usada para criar e testar ambientes de aprendizado por reforço
import random  # Biblioteca para gerar valores aleatórios
from IPython.display import clear_output  # Usado para limpar a saída no Jupyter Notebook
import numpy as np  # Biblioteca para operações matemáticas e manipulação de arrays

# Cria o ambiente do jogo "Taxi-v3" (um simulador onde um táxi deve pegar e deixar passageiros corretamente)
env = gym.make('Taxi-v3', render_mode="ansi")

# Reinicia o ambiente para começar do estado inicial
env.reset()

# Exibe a representação visual do ambiente no terminal
print(env.render())

# Cria a Q-table (tabela que armazena os valores aprendidos para cada ação em cada estado)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Define os hiperparâmetros do algoritmo Q-learning
alpha = 0.1  # Taxa de aprendizado (o quão rápido o agente aprende)
gamma = 0.6  # Fator de desconto (importância das recompensas futuras)
epsilon = 0.1  # Probabilidade de explorar ações aleatórias

# Executa 10.000 episódios de treinamento
for i in range(10000):
    estado = env.reset()  # Reinicia o ambiente a cada episódio

    penalidades, recompensas = 0, 0  # Inicializa penalidades e recompensas
    done = False  # Indica se o episódio terminou

    while not done:  # Continua até o episódio terminar
        # Escolhe entre exploração (ação aleatória) ou exploração (ação da Q-table)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Escolhe uma ação aleatória
        else:
            acao = np.argmax(q_table[estado])  # Escolhe a melhor ação aprendida

        # Executa a ação e recebe o novo estado, recompensa e se o jogo terminou
        proximo_estado, recompensa, done, info = env.step(acao)
