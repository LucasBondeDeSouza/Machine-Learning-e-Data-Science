import gym  # Importa a biblioteca Gym, usada para criar e testar ambientes de aprendizado por reforço
import random  # Biblioteca para gerar valores aleatórios

# Cria o ambiente do jogo "Taxi-v3" (um simulador onde um táxi deve pegar e deixar passageiros corretamente)
env = gym.make('Taxi-v3', render_mode="ansi")

# Reinicia o ambiente para começar do estado inicial
env.reset()

# Exibe a representação visual do ambiente no terminal
print(env.render())

# Exibe as ações possíveis que o táxi pode tomar:
# 0 = Sul, 1 = Norte, 2 = Leste, 3 = Oeste, 4 = Pegar passageiro, 5 = Deixar passageiro
print(env.action_space)

# Exibe o número total de estados possíveis no ambiente (posição do táxi, passageiro e destino)
print(env.observation_space)

# Exibe um dicionário contendo a estrutura de transição do ambiente (probabilidades e recompensas para cada ação)
print(env.P)