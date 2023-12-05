import pandas as pd

# Criando o dataframe de exemplo
dados = {
    'temperatura': [30, 25, 28, 18, 20, 22, 24, 28, 26, 30],
    'umidade': [85, 90, 78, 65, 75, 70, 80, 75, 80, 70],
    'jogar_tenis': ['Não', 'Não', 'Sim', 'Sim', 'Sim', 'Sim', 'Não', 'Sim', 'Sim', 'Não']
}

df = pd.DataFrame(dados)
print(df)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Separando as variáveis de entrada (temperatura e umidade) e o alvo (jogar_tenis)
X = df[['temperatura', 'umidade']]
y = df['jogar_tenis']

# Dividindo o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32, random_state=30)

# Criando o modelo Naive Bayes Gaussiano
modelo = GaussianNB()

# Treinando o modelo
modelo.fit(X_train, y_train)

# Fazendo previsões
y_pred = modelo.predict(X_test)

# Calculando a acurácia do modelo
acuracia = accuracy_score(y_test, y_pred)
print('Acurácia:', acuracia)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=modelo.classes_)
print(cm)
color = 'white'
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo.classes_)
disp.plot()
plt.show()