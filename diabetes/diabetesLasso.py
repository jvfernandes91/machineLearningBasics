print(__doc__)


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Carrega o diabetes dataset
diabetes = datasets.load_diabetes()

# Usa somente o primeiro atributo, a idade
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Separa os dados em dois conjuntos, treino 
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Separa os resultados em conjuntos de treino e teste
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Cria o objeto de Lasso
regr = linear_model.Lasso(alpha=0.1)

# Ajusta o modelo usando os conjuntos de treino
regr.fit(diabetes_X_train, diabetes_y_train)

# Faz as predições usando o conjunto de teste
diabetes_y_pred = regr.predict(diabetes_X_test)

# Calcula o R quadrado(quanto mais próximo de 1, melhor)
print('\nR²: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Calcula o erro quadrático médio(quanto menor, melhor)
print("Erro quadrático médio: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))

# Plota os gráficos
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
plt.xlabel('Idade dos pacientes padronizada')
plt.ylabel('Progressão da doença')

plt.show()