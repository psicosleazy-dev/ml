from ucimlrepo import fetch_ucirepo 
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 

# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
  
# variable information 
#print(heart_disease.variables)

# tratamento dos dados
X = X.fillna(0) #substituindo NaN por 0

def replace_greater_than_zero(value):
    return 1 if value > 0 else value

y_result = y.applymap(replace_greater_than_zero)

y = y.values.flatten() # transformando y de um array 2D para um vetor 1D
y_result = y_result.values.flatten() 

print("DataFrame Original (multiclasse):")
print(y)
print("\nDataFrame Após Substituição (classe binária):")
print(y_result)