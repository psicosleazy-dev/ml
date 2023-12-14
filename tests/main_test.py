from ucimlrepo import fetch_ucirepo 
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 

# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 

X = X.fillna(0)
def replace_greater_than_zero(value):
    return 1 if value > 0 else value

y = y.applymap(replace_greater_than_zero)

y = y.values.flatten()

# area do algoritmo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32, random_state=30)

# Criando o modelo Naive Bayes Gaussiano
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinando o modelo
rf_model.fit(X_train, y_train)

# Fazendo previsões
y_pred = rf_model.predict(X_test)

# fim da area do algoritmo

precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)

print(f'Acurácia: {accuracy}') # taxa de acerto
print(f'Precisão: {precision}') # fracao das predicoes positivas que estavam corretas (exatidao)
print(f'Revocação: {recall}') # taxa de acerto na classe positiva (completude)

cm = confusion_matrix(y_test,y_pred)
# Gerar a matriz de confusão plotada
sn.set(font_scale=1.0) # for label size
sn.heatmap(cm, annot=True, annot_kws={"size": 16},cmap='magma_r') # font size
color = 'white'
plt.show()