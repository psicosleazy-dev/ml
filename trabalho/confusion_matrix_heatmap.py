import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = [[13,1,1,0],
         [3,9,6,0],
         [0,0,16,2],
         [0,0,0,13]]

df_cm = pd.DataFrame(array, range(4), range(4))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.0) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
color = 'white'
plt.show()