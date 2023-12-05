import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix

# Criar um conjunto de dados fictício para exemplo
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o modelo KNN com k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo
knn.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = knn.predict(X_test)

# Calcular a acurácia
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy}')

# Gerar a matriz de confusão plotada
disp = plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
disp.ax_.set_title('Matriz de Confusão Normalizada')
plt.show()