

# Additional Dataset Information
#This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them.  
#In particular, the Cleveland database is the only one that has been used by ML researchers to date. 
# The "goal" field refers to the presence of heart disease in the patient.  It is integer valued from 0 (no presence) to 4. 
#Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).  
#The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.
#One file has been "processed", that one containing the Cleveland database.  All four unprocessed files also exist in this directory.
#To see Test Costs (donated by Peter Turney), please see the folder "Costs" 
#Has Missing Values -> Yes  
# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# Carregando o conjunto de dados de doenças cardíacas
from ucimlrepo import fetch_ucirepo
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

# Obtendo informações sobre o conjunto de dados antes do tratamento
total_rows, total_columns = X.shape
percentage_missing = np.mean(np.isnan(X)) * 100
percentage_positive_cases = np.mean(y) * 100

print(f"The data has {total_rows} rows and {total_columns} columns.")
print(f"The percentage of missing values is: {percentage_missing:.1f}%")
print(f"Percentage of positive cases: {percentage_positive_cases:.1f}%\n")


# Visualizando informações sobre o target (y) no conjunto de dados
class_counts = y['num'].value_counts()

labels = class_counts.index
sizes = class_counts.values

colors = ['green', 'lightgreen', 'yellow', 'lightcoral', 'red']  # Cores para as classes

plt.figure(figsize=(10, 5))

plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, colors=colors)

# Adicionando uma legenda explicativa
legend_labels = ['No presence of heart disease',
                 'Presence (1): Mild heart disease',
                 'Presence (2): Moderate heart disease',
                 'Presence (3): Severe heart disease',
                 'Presence (4): Very severe heart disease']
plt.legend(legend_labels, loc='upper right', bbox_to_anchor=(1.2, 1))

plt.title('Percentage of Presence and No Presence of Heart Disease')
plt.show()

# Inicializando os classificadores com imputação e padronização no pipeline
# kNN
knn_classifier = make_pipeline(
    SimpleImputer(strategy='mean'),  # Imputação de valores ausentes
    StandardScaler(),  # Padronização dos dados
    KNeighborsClassifier()  # Classificador kNN
)

# Árvore de decisão
dt_classifier = make_pipeline(
    SimpleImputer(strategy='mean'),  # Imputação de valores ausentes
    DecisionTreeClassifier()  # Classificador de árvore de decisão
)

# Naive Bayes
nb_classifier = make_pipeline(
    SimpleImputer(strategy='mean'),  # Imputação de valores ausentes
    GaussianNB()  # Classificador Naive Bayes
)

# Lista de classificadores
classifiers = [knn_classifier, dt_classifier, nb_classifier]
classifiers_names = ['kNN', 'Decision Tree', 'Naive Bayes']

# Definindo as métricas de desempenho
scoring_metrics = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

# Configurando a validação cruzada k-fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Dicionário para armazenar os resultados
results = {}

# Loop para avaliar o desempenho de cada classificador
for classifier, name in zip(classifiers, classifiers_names):
    print(f"Validação cruzada {name}...")
    scores = cross_val_score(classifier, X, y.values.ravel(), cv=kf, scoring='accuracy', n_jobs=-1)
    results[name] = scores

# Exibindo os resultados da precisão
print("Resultados da precisão:")
for name in classifiers_names:
    mean_score = np.mean(results[name])
    std_score = np.std(results[name])
    print(f"{name}: Precisão média - {mean_score:.3f}, Desvio padrão - {std_score:.3f}")

# Visualizando os resultados usando boxplot
plt.boxplot([results[name] for name in classifiers_names], labels=classifiers_names)
plt.title("Comparação da precisão")
plt.show()

# Treinando e extraindo a importância das características para kNN e Decision Tree
knn_classifier.fit(X, y.values.ravel())
dt_classifier.fit(X, y.values.ravel())

# Obtendo as importâncias das características
dt_feature_importance = dt_classifier.named_steps['decisiontreeclassifier'].feature_importances_

# Exibindo a importância das características para Decision Tree
plt.figure(figsize=(12, 6))
plt.bar(range(len(dt_feature_importance)), dt_feature_importance, align='center')
plt.xticks(range(len(dt_feature_importance)), X.columns, rotation=45)  # Use X.columns para obter os nomes das características
plt.title('Feature Importance - Decision Tree')
plt.show()


