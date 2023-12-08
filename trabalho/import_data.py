from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 

# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
  
# variable information 
#print(heart_disease.variables)


def replace_greater_than_zero(value):
    return 1 if value > 0 else value

y_result = y.applymap(replace_greater_than_zero)

print("DataFrame Original (multiclasse):")
print(y)
print("\nDataFrame Após Substituição (classe binária):")
print(y_result)