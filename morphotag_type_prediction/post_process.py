from matplotlib import pyplot as plt
import numpy as np
from pandas import *

data_file = open("logs/it-nouns-1l-pg-100it/params.txt","r")
data_lines = data_file.readlines()
gender_lines = data_lines[7:11]

df = DataFrame([[float(s) for s in l.strip().split("\t")[2:]] for l in gender_lines[1:]],\
    index=[l.split("\t")[0] for l in gender_lines[1:]],\
    columns=gender_lines[0].strip().split("\t")[2:])

vals = np.around(df.values,2)

fig = plt.figure(figsize=(40,6))
ax = fig.add_subplot(frameon=True, xticks=[], yticks=[])

the_table=plt.table(cellText=vals, rowLabels=df.index, colLabels=df.columns,\
                    colWidths = [0.03]*vals.shape[1], loc='center',\
                    cellColours=plt.cm.hot(vals))

plt.show()
plt.savefig('it_nouns_pg-gender.png')
