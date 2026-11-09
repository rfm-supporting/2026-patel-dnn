import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('model_matches.txt')

df_ref = df.pivot(columns='model', index='stimname', values='reflum')
df_match = df.pivot(columns='model', index='stimname', values='matchlum')
df_diff = df_ref - df_match
df_diff = df_diff[['odog','retinex99','dakinbex']]
print(df_diff)

df_diff.plot.bar()
plt.ylabel('illusion strength (cd/m$^2$)')
plt.ylim(-10, 20)
plt.show()

# This plot puts the conditions in a different order than in Figure 6
# in the paper, and uses short forms for the condition names. Here is
# a legend for the condition names in this plot.
# 
# (a) painted conditions
# argyle = argyle
# argylel = argyle long
# koffkas = Koffka-Adelson
# koffkat = Koffka broken
# simcon = simultaneous contrast
# simcona = simultaneous contrast articulated
# snake = snake
# checkassim = checkerboard
# white = White's
# 
# (b) cast shadow conditions
# these conditions have the same names as in the painted conditions in
# part (a), but with a capital letter X appended to the name
# 
# (c) control conditions
# argylec = argyle control
# argylelc = argyle long control
# koffkac = Koffka control
# snakec = snake control
