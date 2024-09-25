import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import os

Lista_Neuronios = [100, 500]
Lista_Iterações = [5000, 10000]
Lista_Ativação = ['identity', 'logistic', 'tanh', 'relu']
Rodar_a_partir = 4

for i in range(Rodar_a_partir-1,5):
    print('Carregando Arquivo de teste')
    nome_arquivo_teste = "teste" + str(i + 1) + ".npy"
    arquivo = np.load(nome_arquivo_teste)
    x = arquivo[0]
    y = np.ravel(arquivo[1])

    # Cria uma pasta para o teste específico
    pasta_teste = f'teste_{i + 1}'
    os.makedirs(pasta_teste, exist_ok=True)

    for ativacao in Lista_Ativação:
        for n_iteracoes in Lista_Iterações:
            for n_neuronios in Lista_Neuronios:
                regr = MLPRegressor(hidden_layer_sizes=(n_neuronios),
                                    max_iter=n_iteracoes,
                                    activation=ativacao,
                                    solver='adam',
                                    learning_rate='adaptive',
                                    n_iter_no_change=50)

                print('Treinando RNA')
                regr.fit(x, y)

                print('Preditor')
                y_est = regr.predict(x)

                plt.figure(figsize=[14, 7])

                # plot curso original
                plt.subplot(1, 3, 1)
                plt.plot(x, y)
                plt.title('Curso Original')

                # plot aprendizagem
                plt.subplot(1, 3, 2)
                plt.plot(regr.loss_curve_)
                plt.title('Aprendizagem')

                # plot regressor
                plt.subplot(1, 3, 3)
                plt.plot(x, y, linewidth=1, color='yellow', label='Original')
                plt.plot(x, y_est, linewidth=2, label='Estimado')
                plt.title('Regressor')
                plt.legend()

                # Salva o gráfico na pasta do teste
                nome_grafico = f'grafico_neuronios_{n_neuronios}_iteracoes_{n_iteracoes}_ativacao_{ativacao}.png'
                plt.savefig(os.path.join(pasta_teste, nome_grafico))
                plt.close()  # Fecha a figura para liberar memória

print('Gráficos salvos nas pastas dos testes.')
