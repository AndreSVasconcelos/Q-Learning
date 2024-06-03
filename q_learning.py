# Libs
import numpy as np
import gym
import random

# Criação do ambiente Gym e inicialização de uma nova instância
env = gym.make('Taxi-v3')
env.reset()

# Visualizar o espaço de ações (movimentos possíveis: 0 = south, 1 = north, 2 = east, 3 = west, 4 = pickup, 5 = dropoff)
print(env.action_space)
# Quantidade total de estados possíveis == len(env.P)
print(env.observation_space)
# Observar todas as ações possíveis no ambiente, cada ação possui [probabilidade, proximo_estado, recompensa, info]
#print(env.P)



env.render()