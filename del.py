import sys, os, io, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


from tools import *

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn import metrics

from config import *

from imblearn.under_sampling import RandomUnderSampler

from collections import namedtuple, Counter


## Parameters #########
file_ = 'dataset.xlsx'
workfolder = 'results_delirium'

# 
if not os.path.isdir(workfolder):
  os.mkdir(workfolder)


# select scenario
data_pre = 0
data_pos = 0
data_all = 1

sel1_ = 0
sel2_ = 0
sel3_ = 0
sel_all = 1
##


# final list of features
if sel_all:
  if data_pre:
    features = list(set(features_pre) & set(features_orig))
    label_data = '_vars_pre_sel_none_'
  if data_pos:
    features = list(set(features_pos) & set(features_orig))
    label_data = '_vars_pos_sel_none_'
  if data_all:
    features = list(set(features_orig) & set(features_orig))
    label_data = '_vars_all_sel_none_'
elif sel1_:
  if data_pre:
    features = list(set(features_pre) & set(features_sel1))
    label_data = '_vars_pre_sel1_'
  if data_pos:
    features = list(set(features_pos) & set(features_sel1))
    label_data = '_vars_pos_sel1_'
  if data_all:
    features = list(set(features_orig) & set(features_sel1))
    label_data = '_vars_all_sel1_'
elif sel2_:
  if data_pre:
    features = list(set(features_pre) & set(features_sel2))
    label_data = '_vars_pre_sel2_'
  if data_pos:
    features = list(set(features_pos) & set(features_sel2))
    label_data = '_vars_pos_sel2_'
  if data_all:
    features = list(set(features_orig) & set(features_sel2))
    label_data = '_vars_all_sel2_'
elif sel3_:
  if data_pre:
    features = list(set(features_pre) & set(features_sel3))
    label_data = '_vars_pre_sel3_'
  if data_pos:
    features = list(set(features_pos) & set(features_sel3))
    label_data = '_vars_pos_sel3_'
  if data_all:
    features = list(set(features_orig) & set(features_sel3))
    label_data = '_vars_all_sel3_'
else:
  print ('problema')
  sys.exit(0)


n_jobs = 6 # number of independent processes for nested cross-validation

cv = 3
write_figs = 1
pdp = 1
report_nested_cross_hypert_tuning = 1
n_bootstraps = 2

####################################





buf = io.StringIO()

# abrindo arquivo
df = pd.read_excel(file_)[features]
df_a = df.copy()


# remove lines with output NaN
df_a = df_a[(~df_a[y_name].isna())]


# randomize
df_a = df_a.sample(frac=1, random_state=int(time.time()))
df_a = df_a.reset_index(drop=True)


# copy
df = df_a.copy()



print (df_a.columns)


info_nan_txt = ''


df_x = df_a[df_a[y_name]==0]
df_y = df_a[df_a[y_name]==1]

registros_y_0 = len(df_x)
registros_y_1 = len(df_y)

s = ''
s = s + 'Antes\n'
s = s + 'Registros com yname == 1: {}\n'.format(registros_y_1)
s = s + 'Registros com yname == 0: {}\n'.format(registros_y_0)
s = s + '#nans em cada record com yname=1\n'
s = s + str(Counter(np.isnan(df_y).sum(axis=1))) + '\n'
s = s + '#nans em cada record com yname=0\n'
s = s + str(Counter(np.isnan(df_x).sum(axis=1))) + '\n'
s = s + '\n'

df_x_antes = df_x.copy()
df_y_antes = df_y.copy()

###

df_y, cols = eliminate_cols_nan(df_y, 0.20)
s = s + f'colunas eliminadas: {cols}\n'
print (s)
df_x = df_x.drop(columns=cols)
  
df_y, records = eliminate_records_nan(df_y, 0.9, reset_index=True)
df_x, records = eliminate_records_nan(df_x, 0, reset_index=True)


n_registros_y_1 = len(df_y)
n_registros_y_0 = len(df_x)

s = s + 'Depois\n'
s = s + 'Registros com yname == 1: {}\n'.format(n_registros_y_1)
s = s + 'Registros com yname == 0: {}\n'.format(n_registros_y_0)


df_a = pd.concat([df_x, df_y])
df_a = df_a.sample(frac=1, random_state=int(time.time()))
df_a = df_a.reset_index(drop=True)

s = s + 'df_a.shape: {}\n'.format(df_a.shape)

s = s + 'Colunas eliminadas:\n'
s = s + str(cols) + '\n'


s = s + '#nans em cada record com yname=1\n'
s = s + str(Counter(np.isnan(df_y).sum(axis=1))) + '\n'

s = s + '#nans em cada record com yname=0\n'
s = s + str(Counter(np.isnan(df_x).sum(axis=1))) + '\n'

info_nan_txt = s

df_x_depois = df_x.copy()
df_y_depois = df_y.copy()




lst_data_bstrap = list()

rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(df_a, df_a[y_name].values)

if n_bootstraps > 1:
  for i in range(n_bootstraps):
    X_resampled, y_resampled = rus.fit_resample(df_a, df_a[y_name].values)
    lst_data_bstrap.append(pd.DataFrame(X_resampled, columns=df_a.columns))

# define df_a as the last bootstrap dataset    
df_a = pd.DataFrame(X_resampled, columns=df_a.columns)         

#

file_ = open(os.path.join(workfolder, 'info_nans.txt'), 'w')
file_.write(info_nan_txt)
file_.close()

###






## PDP
if pdp:
  lst_features = list(df.columns)
  lst_features.remove(y_name)
  
  try:
    pdp_dataset(df, lst_features, y_name, n=4, writefolder=workfolder)
  except:
    print ('Problem in pdp function.')






# write df info 
df_a.info(buf=buf)
s = buf.getvalue()
file_ = open(os.path.join(workfolder, 'info.txt'), 'w')
file_.write(s)
file_.close()





# columns with 1 unique value will be eliminated
lst_vars_to_drop = []
for var in df_a.columns:
  if len(df_a[var].value_counts()) == 1:
    lst_vars_to_drop.append(var)
if len(lst_vars_to_drop)>0:
  print ('Drop colunas: {}'.format(lst_vars_to_drop))
  df_a = df_a.drop(columns=lst_vars_to_drop)

# bootstrap case
if n_bootstraps > 1:
  for df_temp in lst_data_bstrap:

    lst_vars_to_drop = []
    for var in df_temp.columns:
      if len(df_temp[var].value_counts()) == 1:
        lst_vars_to_drop.append(var)
    if len(lst_vars_to_drop)>0:
      print ('Drop colunas: {}'.format(lst_vars_to_drop))
      df_temp = df_temp.drop(columns=lst_vars_to_drop)
    


# standardize 
try:
  df2 = (df_a-df_a.mean())/df_a.std()
  df2 = df2.dropna()
except:
  print ('Checar se nao tem uma variavel com um soh valor.')
  sys.exit(0)

# caso do bootstrap
if n_bootstraps > 1:
  for i in range(n_bootstraps):
  
    # clear nans
    lst_data_bstrap[i] = lst_data_bstrap[i].dropna()
  
    # standardize 
    try:
      lst_data_bstrap[i] = (lst_data_bstrap[i]-lst_data_bstrap[i].mean())/lst_data_bstrap[i].std()
    except:
      print ('Checar se nao tem uma variavel com um soh valor.')
      sys.exit(0)


  
# data standard
df_X = df2.drop([y_name], axis=1)
df_Y = np.rint(df2.copy()[y_name])

# caso boostrap
lst_df_X_bstrap = list()
lst_df_Y_bstrap = list()

if n_bootstraps > 1:
  for i, df_temp in enumerate(lst_data_bstrap):
    lst_df_X_bstrap.append(df_temp.drop([y_name], axis=1))
    lst_df_Y_bstrap.append(np.rint(df_temp.copy()[y_name]))
    

#
list_best_models = list()
l_ = list()

if report_nested_cross_hypert_tuning:

  if n_bootstraps > 1:
    for i in range(n_bootstraps):

      # bootstrap
      #df_X_temp = df_X.sample(frac=1, replace=True)
      #df_Y_temp = df_Y[df_X_temp.index].copy()

      list_best_models, auc_scores_holdout, (fpr_, tpr_) = grid_search_nested_parallel(lst_df_X_bstrap[i].values, 
                                                                                       lst_df_Y_bstrap[i].values, 
                                                                                       cv=3, 
                                                                                       writefolder=workfolder, 
                                                                                       n_jobs=n_jobs, 
                                                                                       resampling=None, 
                                                                                       roc_curve_output=True, 
                                                                                       standardize=False, 
                                                                                       bootstrap=True,
                                                                                       )

      print (auc_scores_holdout)
      l_.append(auc_scores_holdout)

    std_ = np.array(l_).mean(axis=1).std()
    mean_ = np.array(l_).mean(axis=1).mean()

  elif n_bootstraps == 1:
    list_best_models, auc_scores_holdout, (fpr_, tpr_) = grid_search_nested_parallel(df_X.values, df_Y.values, cv=3, writefolder=workfolder, n_jobs=n_jobs, resampling=None, roc_curve_output=True)
    print (auc_scores_holdout)
    l_.append(auc_scores_holdout)


file_ = open(os.path.join(workfolder, 'auc_scores_holdouts.txt'), 'w')
if n_bootstraps > 1:
  file_.write('mean: '+str(mean_)+'\n')
  file_.write('std: '+str(std_)+'\n')
file_.write(str(l_))
file_.close()

if n_bootstraps > 1:
  filepath = os.path.join(workfolder, 'data_roc_curve.txt')
  np.savetxt(filepath, np.stack((fpr_, tpr_)).T, fmt='%s', header=str(features))


# Ensemble model
estimators_list = [(list_best_models[x]+'/'+str(x+1), get_model_ml_(list_best_models[x])) for x in range(3)]
eclf1 = VotingClassifier(estimators=estimators_list, voting='soft')
eclf1 = eclf1.fit(df_X.values, df_Y.values)
  


##### REPORTS ##
# best nested reports
if len(list_best_models)>0:
  for params in list_best_models:
    general_model_report(params, df_X.values, df_Y.values, write_folder=workfolder, cv=cv, balanced=False, labels=df_X.columns, augmented=None)


print ('Run Finished. Workfolder: {}.'.format(workfolder))



# ROC curve for this run, you can aggregate more as in the commented example, this can be used as another script

plt.figure(figsize=(9, 7))

plt.style.use('style_fig.mplstyle')


files_ = [
#  '1_all_data_roc_curve.txt',
#  '2_pre_all_data_roc_curve.txt',
#  '3_pos_all_data_roc_curve.txt',
#  '4_pre_pos_sel1_data_roc_curve.txt',
#  '5_pre_pos_sel3_data_roc_curve.txt',
  'data_roc_curve.txt',
]

labels = [
  #'All pre features',
  #'All pos features',
  #'All pre and pos features',
  #'Pre and pos features selected with PDP',
  #'Age, number of complications, \npreoperative length of stay features',
  'scenario description',
]

stds = [
  #'sd = 0.06',
  #'sd = 0.04',
  #'sd = 0.05',
  #'sd = 0.03',
  'sd = 0.04', # must calculate from auc_scores_holdout.txt

]

m_list = list()
for file_ in files_:
  m_list.append(np.loadtxt(os.path.join(workfolder, file_)))

auc_list = list()
for m in m_list:
  auc_list.append(metrics.auc(m[:, 0], m[:, 1]))


for m, label, auc_, std in zip(m_list, labels, auc_list, stds):
  plt.plot(m[:, 0], m[:, 1], '-', label=f'{label}, AUC={auc_:.2f}, {std}', linewidth=5)

plt.plot([0, 1], [0, 1], '--', label='Chance', color='gray')
plt.plot([0, 1], [0.9, 0.9], '-.', color='gray', linewidth=2.0)
plt.text(0.15, 0.92, 'Sensitivity=0.9')

plt.legend(loc='lower right')

plt.xlabel('1-Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate) ')

#plt.show()
plt.tight_layout()
plt.savefig(os.path.join(workfolder, 'roc_curve.png'), dpi=280)



### Probability map when only age, num_complications and tempo_int_preop in features list

if 'Idade' in features and 'Num_Complicacoes' in features and 'Tempo_Int_Preop' in features and len(features)==3:

        tempo_col = 'Tempo_Int_Preop'
        #cmap = 'plasma_r'
        #cmap = 'inferno_r'
        cmap = 'magma_r'
        #cmap = 'gnuplot_r'
        #cmap = 'gnuplot2_r'
        #cmap = 'CMRmap_r'

        #tempo_v = np.linspace(0, 100, 40)
        tempo_v = np.linspace(1, 10, 40)
        nc_v = np.linspace(0, 10, 40)

        age_par1 = 55
        age_par2 = 75
        age_par3 = 85
        #vmax = 0.07
        #vmin = 0.


        xx, yy = np.meshgrid(tempo_v, nc_v)


        # age 1
        input_list = list()
        for t, n in zip(xx.ravel(order='C'), yy.ravel(order='C')):
          input_list.append([(age_par1-df_a['Idade'].mean())/df_a['Idade'].std(),
                        (n-df_a['Num_Complicacoes'].mean())/df_a['Num_Complicacoes'].std(),
                        (t-df_a[tempo_col].mean())/df_a[tempo_col].std(),
                        ])


        #plist = eclf1.predict_proba(np.array(input_list).reshape(1, -1)) 
        plist = eclf1.predict_proba(np.array(input_list)) 
        plist = np.array(plist)*(117./(1336+117))
        plist = np.squeeze(plist)[:, 1]
        plist = plist.reshape(xx.shape, order='C')

        # age 2
        input_list = list()
        for t, n in zip(xx.ravel(order='C'), yy.ravel(order='C')):
          input_list.append([(age_par2-df_a['Idade'].mean())/df_a['Idade'].std(),
                        (n-df_a['Num_Complicacoes'].mean())/df_a['Num_Complicacoes'].std(),
                        (t-df_a[tempo_col].mean())/df_a[tempo_col].std(),
                        ])


        #plist = eclf1.predict_proba(np.array(input_list).reshape(1, -1)) 
        plist2 = eclf1.predict_proba(np.array(input_list)) 

        plist2 = np.array(plist2)*(117./(1336+117))
        plist2 = np.squeeze(plist2)[:, 1]
        plist2 = plist2.reshape(xx.shape, order='C')


        # age 3
        input_list = list()
        for t, n in zip(xx.ravel(order='C'), yy.ravel(order='C')):
          input_list.append([(age_par3-df_a['Idade'].mean())/df_a['Idade'].std(),
                        (n-df_a['Num_Complicacoes'].mean())/df_a['Num_Complicacoes'].std(),
                        (t-df_a[tempo_col].mean())/df_a[tempo_col].std(),
                        ])


        #plist = eclf1.predict_proba(np.array(input_list).reshape(1, -1)) 
        plist3 = eclf1.predict_proba(np.array(input_list)) 

        plist3 = np.array(plist3)*(117./(1336+117))
        plist3 = np.squeeze(plist3)[:, 1]
        plist3 = plist3.reshape(xx.shape, order='C')



        #
        vmax = max(plist.max(), plist2.max())
        vmin = max(plist.min(), plist2.min())


        fig, axs = plt.subplots(1, 2, figsize=(48, 20), dpi=100)

        ax = axs[0]
        c1 = ax.imshow(plist,
                   origin='lower',
                   interpolation='nearest',
                   aspect='auto',
                   extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                   vmin=vmin, vmax=vmax,
                   cmap=cmap,
                   )
        ax.set_xlabel('Preoperative length of hospital stay')
        ax.set_ylabel('Total Number of Complications')
        ax.set_title('Age: '+str(age_par1))
        cbar = fig.colorbar(c1, ax=axs[0])
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("Delirium probability of ocurrence", rotation=270)

        ax = axs[1]
        c2 = ax.imshow(plist2,
                   origin='lower',
                   interpolation='nearest',
                   aspect='auto',
                   extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                   vmin=vmin, vmax=vmax,
                   cmap=cmap,
                   )
        ax.set_xlabel('Preoperative length of hospital stay')
        #ax.set_ylabel('Total Number of Complications')
        ax.set_title('Age: '+str(age_par2))
        cbar = fig.colorbar(c2, ax=axs[1])
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("Delirium probability of ocurrence", rotation=270)


        #fig.tight_layout()


        #plt.show()
        plt.savefig(os.path.join(workfolder, 'probability_map.png'))


