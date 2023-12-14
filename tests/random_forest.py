import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Criar um conjunto de dados fictício para exemplo
X, y = make_classification(
    n_samples=100,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_clusters_per_class=1,
    random_state=42
)

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32, random_state=30)

# Inicializar o modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinar o modelo
rf_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = rf_model.predict(X_test)

# Avaliar a performance do modelo
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)

print(f'Acurácia: {accuracy}') # taxa de acerto
print(f'Precisão: {precision}') # fracao das predicoes positivas que estavam corretas (exatidao)
print(f'Revocação: {recall}') # taxa de acerto na classe positiva (completude)

cm = confusion_matrix(y_test,y_pred)

# Gerar a matriz de confusão plotada
df_cm = pd.DataFrame(cm, range(2), range(2))
sn.set(font_scale=1.0) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
color = 'white'
plt.show()