print(__doc__)

import matplotlib.pyplot as plt
from sklearn import datasets, tree, metrics

#O digits dataset
digits = datasets.load_digits()

#Mostra as quatro primeiras imagens armazenadas no atributo 'images'
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

#Transforma os dados em uma matrix (samples, feature)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

#Cria o objeto de classificação da Decision Tree
clf = tree.DecisionTreeClassifier()

#Os dígitos da primeira metade dos dados são aprendidos
clf.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

#A segunda metade é usada para a predição
expected = digits.target[n_samples // 2:]
predicted = clf.predict(data[n_samples // 2:])

#Mostra a acurácia do modelo utilizado
print("\nAcurácia para o classificador Decision Tree: %.2f\n" 
      % (metrics.accuracy_score(expected, predicted)))

#Mostra os quatro primeiros dígitos do conjunto de predições e as predições
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()