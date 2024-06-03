# Libs
from IPython.display import clear_output
from time import sleep
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

# Parametros
alpha = 0.1 # Taxa de aprendizagem
gamma = 0.6 # Taxa de desconto
epsilon = 0.1 # Taxa de exploração - exploration/exploitation
q_table = np.zeros([env.observation_space.n, env.action_space.n]) # Criação da tabela Q

# Roda o Ambiente
for i in range(10000):
    estado = env.reset()
    penalidades, recompensa = 0, 0
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            acao = env.action_space.sample()
        else:
            acao = np.argmax(q_table[estado])
        
        proximo_estado, recompensa, done, info = env.step(acao)
        q_antigo = q_table[estado, acao]
        proximo_maximo = np.max(q_table[proximo_estado])
        q_novo = (1 - alpha) * q_antigo + alpha * (recompensa + gamma * proximo_maximo)
        q_table[estado][acao] = q_novo

        if recompensa == -10:
            penalidades += 1

        estado = proximo_estado
    if i % 100 == 0:
        clear_output(wait=True)
        print(f'Episodio: {i}')


# Visualizar o resultado
print('Treinamento concluído!')
#print(q_table)

total_penalidades = 0
episodios = 50
frames = []

for _ in range(episodios):
    estado = env.reset()
    penalidades, recompensa = 0, 0
    done = False
    while not done:
        acao = np.argmax(q_table[estado])
        estado, recompensa, done, info = env.step(acao)
        if recompensa == -10:
            penalidades += 1
        
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': estado,
            'action': acao,
            'reward': recompensa
        })
    total_penalidades += penalidades

print(f'episodios: {episodios}, penalidades: {total_penalidades}')

for frame in frames:
    clear_output(wait=True)
    print(frame['frame'])
    print(f"Timestep: {frame['state']}")
    print(f"Action: {frame['action']}")
    print(f"Reward: {frame['reward']}")
    sleep(.1)