import pickle
import seaborn as sns; sns.set_theme()
with open('results_task4.pkl', 'rb') as f:
    res = pickle.load(f)

import ipdb;  ipdb.set_trace()
# for i in range(0,6):
i = 2
# import ipdb; ipdb.set_trace()
# xticklabels=['L2Reg','MP', 'OMP','BP','L0','ADMM']
ax = sns.heatmap(res[:,:,i]-res[0,0,i], linewidths=.1, xticklabels=['N0','N1', 'N2','N3','N4'], yticklabels=['N0','N1', 'N2','N3','N4'], cmap="YlGnBu")
ax.get_figure().savefig('test'+str(i))