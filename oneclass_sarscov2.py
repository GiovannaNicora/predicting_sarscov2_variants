from change_point_detection.reliability_utils import *
from change_point_detection.loading_data import *
from collections import Counter
import matplotlib.pyplot as plt
from datetime import *
import logging
from sklearn.model_selection import ParameterGrid
from sklearn.svm import OneClassSVM

def get_time(date_string, date_format= "%Y-%m-%d"):
    return datetime.strptime(date_string, date_format).date()

"""

Reading files 

"""
""" When each variant has been classified as VOI or VOC"""
var_class_time = pd.read_csv('/mnt/resources/2022_04/2022_04/sars_cov2_time_classification.txt', sep='\t')
retraining_week = [27, 35, 45, 48, 49, 51, 62, 75]
# header
header = pd.read_csv('/mnt/resources/2022_04/2022_04/dataset_week/1/EPI_ISL_489939.csv', nrows=1)

"""
Useful variable

"""

# columns in metadata
col_class_variant = 'Variant'
col_submission_date = 'Collection.date'
col_variant_id = 'Accession.ID'


non_neutral_variants = metadata[col_class_variant].unique().tolist()
non_neutral_variants.remove('unknown')
# kmers features
features = header.columns[1:].tolist()

date_format = "%Y-%m-%d"

# datetime.strptime(var_class_time.loc['Alpha'].values[0], date_format).date() < datetime.strptime(var_class_time.loc['Delta'].values[0], date_format).date()
# get_time(var_class_time.loc['Alpha'].values[0]) < get_time(metadata[col_submission_date].iloc[0])

logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler('/mnt/resources/2022_04/2022_04/'+'run_main_oneclass_retrain_tmp.log', 'w+'),
                            logging.StreamHandler()
                        ])

# Training week
starting_week = 1


# Loading first training step
df_trainstep_1, train_w_list = load_data(dir_week, [starting_week])
train_step1 = df_trainstep_1.iloc[:, 1:len(df_trainstep_1.columns)].to_numpy()

sum_train = np.sum(train_step1, axis=0)
i_no_zero = np.where(sum_train != 0)[0]
y_train_initial = metadata[metadata[col_variant_id].isin(df_trainstep_1.iloc[:, 0].tolist())][col_class_variant]

y_train_class = map_variant_to_finalclass(y_train_initial.tolist(), non_neutral_variants)
counter_i = Counter(y_train_initial) # at the beginning, all the variants were "unknown"=neutral



# filtering out features with all zero
train_step1 = train_step1[:, i_no_zero]

# training one class
p_grid = {'kernel': ['rbf'], 'gamma':['scale', 'auto'], 'nu':[0.01, 0.2, 0.5, 0.7, 0.85, 1],
          'shrinking':[True, False], }

all_combo = list(ParameterGrid(p_grid))

results_fine_tune = []
for combo in all_combo[0:1]:
    combo
    logging.info("---> One Class SVM - Param: "+str(combo))
    clf_c = OneClassSVM(**combo)
    clf_c.fit(train_step1)

    y_test_dict_variant_type = {}
    y_test_dict_finalclass = {}
    y_test_dict_predictedclass = {}

    train = train_step1.copy()
    for week in range(1,  metadata['week'].max()):
         if week in retraining_week:
            logging.info('----> RETRAINING <-----')
            clf_c = OneClassSVM(**combo)
            clf_c.fit(train)

         week_date_list = list(set(metadata[metadata['week']==starting_week+week][col_submission_date].tolist()))
         week_date = str(min([get_time(x) for x in week_date_list]))
         logging.info("# Week "+str(starting_week+week)+ "| Week date:"+week_date)
         # Loading first test step
         df_teststep_i, test_w_list = load_data(dir_week, [starting_week+week])
         test_step_i = df_teststep_i.iloc[:, 1:len(df_teststep_i.columns)].to_numpy()
         test_step_i = test_step_i[:, i_no_zero]
         y_test_step_i = get_variant_class(metadata, df_teststep_i.iloc[:, 0].tolist())
         y_test_dict_variant_type[starting_week+week] = y_test_step_i
         y_test_fclass_i = map_variant_to_finalclass(y_test_step_i, non_neutral_variants)
         i_voc = np.where(np.array(y_test_fclass_i)==-1)[0]
         y_test_dict_finalclass[starting_week+week] = y_test_fclass_i
         variant_dict = Counter(y_test_step_i)

         # predict
         # Returns -1 for outliers and 1 for inliers.
         y_test_i_predict = clf_c.predict(test_step_i)
         i_inlier = np.where(y_test_i_predict==1)[0]
         inlier_test_i = test_step_i[i_inlier]
         train = np.concatenate((train, inlier_test_i))


         y_test_dict_predictedclass[starting_week + week] = y_test_i_predict
         y_test_voc_predict = np.array(y_test_i_predict)[i_voc]


         logging.info("Number of variants in week:"+str(test_step_i.shape[0]))
         logging.info("Number of variants of concern in week:"+str(len([x for x in y_test_fclass_i if x == -1])))
         logging.info("Distribution of variants of concern:" + str(Counter(y_test_step_i)))
         logging.info("Number of variants predicted as anomalty:"+str(len([x for x in y_test_dict_predictedclass[starting_week+week] if x==-1])))
         acc_voc = len([x for x in y_test_voc_predict if x==-1])
         logging.info("Number of VOC variants predicted as anomalty:"+str(acc_voc))

         for k in variant_dict.keys():
             i_k = np.where(np.array(y_test_step_i)==k)[0]
             logging.info('Number of '+k+' variants:'+str(len(i_k))+'; predicted anomalty='+str(len([x for x in y_test_i_predict[i_k] if x == -1])))

    # saving results for this comb of param of the oneclass_svm
    results = {'y_test_variant_type': y_test_dict_variant_type,
               'y_test_final_class':y_test_dict_finalclass,
               'y_test_predicted_class':y_test_dict_predictedclass}
    results_fine_tune.append(results)

y_true_model0 = results_fine_tune[0]['y_test_final_class']
y_predict_model0 = results_fine_tune[0]['y_test_predicted_class']

fp_list = []
n_list = []
fn_list = []
n_outlier_list = []

for k in y_true_model0.keys():
    yt = np.array(y_true_model0[k])
    yp = np.array(y_predict_model0[k])

    i_inlier = np.where(yt == 1)[0]
    n_fp = len(np.where(yp[i_inlier] == -1)[0])

    fp_list.append(n_fp)
    n_list.append(len(i_inlier))

    i_outlier = np.where(yt == -1)[0]
    n_fn = len(np.where(yp[i_outlier] == 1)[0])
    fn_list.append(n_fn)
    n_outlier_list.append(len(i_outlier))

tn_list = []
tp_list = []

prec_list = []
recall_list = []
spec_list = []
f1_list = []
for i in range(len(fp_list)):
    tp = n_outlier_list[i]-fn_list[i]
    tn = n_list[i]-fp_list[i]
    tn_list.append(tn)
    tp_list.append(tp)
    if tp + fp_list[i] != 0:
        prec = tp / (tp+fp_list[i])
    else:
        prec = 0

    if tp+fn_list[i] != 0:
        rec = tp / (tp+fn_list[i])
    else:
        rec = 0

    if tn+fp_list[i] !=0:
        spec = tn /(tn+fp_list[i])
    else:
        spec = 0

    if prec + rec != 0:
        f1 = 2*prec*rec/(prec+rec)
    else:
        f1 = 0
    f1_list.append(f1)
    spec_list.append(spec)
    prec_list.append(prec)
    recall_list.append(rec)

df_conf = pd.DataFrame()
df_conf['TN'] = tn_list
df_conf['FP'] = fp_list
df_conf['FN'] = fn_list
df_conf['TP'] = tp_list
df_conf['Precision'] = prec_list
df_conf['Recall'] = recall_list
df_conf['F1'] = f1_list
df_conf['Specificity'] = spec_list

df_conf.to_csv('/mnt/resources/2022_04/2022_04/conf_mat_over_time.tsv', sep='\t', index=None)


x = np.arange(len(fp_list))
fig, ax = plt.subplots(figsize=(32,14))
plt.rcParams.update({'font.size':18})
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
r = plt.bar(x, prec_list,  width=0.35, alpha=0.8, color='#a32b15')
# plt.bar_label(r, rotation=0, fontsize=16)
plt.yticks(fontsize=25)
plt.xticks(x, labels=[str(y+2) for y in x], rotation=45, fontsize=20)
plt.xlabel('Week', fontsize=25)

plt.title('Precision in time from 2020-07-06 to 2022-03-23', fontsize=25)
plt.savefig('/mnt/resources/2022_04/2022_04/precision_in_time.png', dpi=350)

fig, ax = plt.subplots(figsize=(32,14))
plt.rcParams.update({'font.size':18})
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
r = plt.bar(x, recall_list,  width=0.35, alpha=0.8, color='#a32b15')
# plt.bar_label(r, rotation=0, fontsize=16)
plt.yticks(fontsize=25)
plt.xticks(x, labels=[str(y+2) for y in x], rotation=45, fontsize=20)
plt.xlabel('Week', fontsize=25)

plt.title('Recall in time from 2020-07-06 to 2022-03-23', fontsize=25)
plt.savefig('/mnt/resources/2022_04/2022_04/recall_in_time.png', dpi=350)

fig, ax = plt.subplots(figsize=(32,14))
plt.rcParams.update({'font.size':18})
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
r = plt.bar(x, f1_list,  width=0.35, alpha=0.8, color='#a32b15')
# plt.bar_label(r, rotation=0, fontsize=16)
plt.yticks(fontsize=25)
plt.xticks(x, labels=[str(y+2) for y in x], rotation=45, fontsize=20)
plt.xlabel('Week', fontsize=25)

plt.title('F1 in time from 2020-07-06 to 2022-03-23', fontsize=25)
plt.savefig('/mnt/resources/2022_04/2022_04/f1_in_time.png', dpi=350)

x = np.arange(len(fp_list))

fig, ax = plt.subplots(figsize=(32,14))
# plt.rcParams.update({'font.size':18})
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
r = plt.bar(x, fp_list,  width=0.35, alpha=0.8, color='#a32b15')
new_datalab = [str(x) for x in r.datavalues]
for i in range(len(new_datalab)):
    if i % 2 != 0:
        new_datalab[i] = ''
# plt.bar_label(r, new_datalab, fontsize=25, padding=2)
plt.yticks(fontsize=30)
newlab_x = []
for i in range(len(x)):
    if i % 5 == 0:
        newlab_x.append(str(x[i]+2))
    else:
        newlab_x.append('')

# plt.xticks(x, labels=[str(y+2) for y in x], rotation=45, fontsize=20)
plt.xticks(x, labels=newlab_x, rotation=0, fontsize=28)
plt.xlabel('Week', fontsize=25)
plt.grid(axis='y')
plt.title('Number of False Positive from 2020-07-06 to 2022-03-23', fontsize=30)
plt.tight_layout()
plt.savefig('/mnt/resources/2022_04/2022_04/n_fp_v3.png', dpi=350)


perc_fp = []
for i in range(len(fp_list)):
    if n_list[i] != 0:
        perc_fp.append(round(fp_list[i]/n_list[i], 2))
    else:
        perc_fp.append(0)
x = np.arange(len(fp_list))
fig, ax = plt.subplots(figsize=(32,14))
plt.rcParams.update({'font.size':18})
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
r = plt.bar(x, perc_fp,  width=0.35, alpha=0.8, color='#a32b15')
plt.bar_label(r, rotation=0, fontsize=16)
plt.yticks(fontsize=25)
plt.xticks(x, labels=[str(y+2) for y in x], rotation=45, fontsize=20)
plt.xlabel('Week', fontsize=25)

plt.title('Percentage of False Positive from 2020-07-06 to 2022-03-23', fontsize=25)
plt.savefig('/mnt/resources/2022_04/2022_04/perc_fp.png', dpi=350)



""" Plotting number of variants """
week_keys = results_fine_tune[0]['y_test_final_class'].keys()
count_neutral = []
count_nonneutral = []
predicted_neural = []
predicted_nonneutral = []

for w in week_keys:
    d = Counter( results_fine_tune[0]['y_test_final_class'][w])
    dp = Counter( results_fine_tune[0]['y_test_predicted_class'][w])

    if 1 in d.keys():
        count_neutral.append(d[1])
    else:
        count_neutral.append(0)

    if -1 in d.keys():
        count_nonneutral.append(d[-1])
    else:
        count_nonneutral.append(0)
    if 1 in dp.keys():
        predicted_neural.append(dp[1])
    else:
        predicted_neural.append(0)

    if -1 in dp.keys():
        predicted_nonneutral.append(dp[-1])
    else:
        predicted_nonneutral.append(0)

perc_out = []
for i in range(len(predicted_neural)):
    perc_out.append(predicted_nonneutral[i]/(predicted_neural[i]+predicted_nonneutral[i]))

x = np.arange(len(count_nonneutral))
w=0.35

fig, ax = plt.subplots(figsize=(32,14))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.rcParams.update({'font.size': 12})
plt.bar([y-w*0.5 for y in x], count_neutral, label='Neutral', width=w)
plt.bar([y+w*0.5 for y in x], count_nonneutral, label='VOC/VOI', width=w)
plt.vlines(x[0], 0, 8000, linestyle = ':')
plt.text(x[0], 8000, '2020-07-06', fontsize=20)
plt.vlines(x[len(x)-1], 0, 8000, linestyle = ':')
plt.text(x[len(x)-1], 8000, '2022-03-23', fontsize=20)
heights = [9000, 9000, 8500, 10500, 9500, 8000, 9000, 10000]
vtype= ['Alpha\nBeta\nGamma', 'Epsil\nIota\nZeta', 'Kappa',
        'Theta', 'Lambda', 'Delta', 'Mu', 'Omicron']
for i,rw in enumerate(retraining_week):
    week_date_list = list(set(metadata[metadata['week'] == rw][col_submission_date].tolist()))
    week_date = str(min([get_time(x) for x in week_date_list]))
    plt.vlines(rw-2, 0, heights[i],
               #transform=ax.get_xaxis_transform(),
               linestyle='-', color='red')
    plt.text(rw-2, heights[i], week_date+'\n'+vtype[i], rotation=0, fontsize=20)

plt.yticks(fontsize=30)
#plt.xticks(x, labels=[str(y+2) for y in x], rotation=45, fontsize=20)
plt.xticks(x, labels=newlab_x, rotation=0, fontsize=28)
plt.xlabel('Week', fontsize=25)
plt.legend(fontsize=25)
plt.title('Distribution of Non-neutral variants in GISAID from 2020-07-06 to 2022-03-23', fontsize=25)
plt.savefig('/mnt/resources/2022_04/2022_04/distribution_ground_truth_v2.png', dpi=350)



x = np.arange(len(count_nonneutral))
w=0.35

fig, ax = plt.subplots(figsize=(32,14))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.rcParams.update({'font.size': 12})
plt.bar([y-w*0.5 for y in x], predicted_neural, label='Predicted Neutral', width=w, alpha=0.8)
plt.bar([y+w*0.5 for y in x], predicted_nonneutral, label='Predicted anomaly', width=w, alpha=0.8)
plt.vlines(x[0], 0, 8000, linestyle = ':')

plt.text(x[0], 8000, '2020-07-06', fontsize=20)
plt.vlines(x[len(x)-1], 0, 8000, linestyle = ':')
plt.text(x[len(x)-1], 8000, '2022-03-23', fontsize=20)
heights = [9000, 9000, 8500, 10500, 9500, 8000, 9000, 10000]

vtype= ['Alpha\nBeta\nGamma', 'Epsil\nIota\nZeta', 'Kappa',
        'Theta', 'Lambda', 'Delta', 'Mu', 'Omicron']
for i,rw in enumerate(retraining_week):
    week_date_list = list(set(metadata[metadata['week'] == rw][col_submission_date].tolist()))
    week_date = str(min([get_time(x) for x in week_date_list]))
    plt.vlines(rw-2, 0, heights[i],
               #transform=ax.get_xaxis_transform(),
               linestyle='-', color='red')
    plt.text(rw-2, heights[i], week_date+'\n'+vtype[i], rotation=0, fontsize=20)

# plt.xticks(x, labels=[str(y+2) for y in x], rotation=45)
plt.xticks(x, labels=newlab_x, rotation=0, fontsize=28)
plt.yticks(fontsize=30)
#plt.xticks(x, labels=[str(y+2) for y in x], rotation=45, fontsize=20)
# plt.xticks(x, labels=newlab_x, rotation=0, fontsize=28)
plt.xlabel('Week', fontsize=25)
plt.legend(fontsize=25, loc='upper left')
plt.xlabel('Week')
plt.title('Distribution of predictions from 2020-07-06 to 2022-03-23', fontsize=25)
plt.savefig('/mnt/resources/2022_04/2022_04/distribution_pred.png', dpi=350)

""" Percentage pof predicted as anomaly"""

perc_nonneutral = []
for i in range(len(predicted_nonneutral)):
    perc_nonneutral.append(predicted_nonneutral[i]/(predicted_nonneutral[i]+predicted_neural[i]))

fig, ax = plt.subplots(figsize=(32,14))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.rcParams.update({'font.size': 12})
plt.bar(x, perc_nonneutral,  width=w, alpha=0.8, color='orange')
plt.vlines(x[0], 0, 0.1, linestyle = ':')
plt.text(x[0], 0.1, '2020-07-06', fontsize=20)
plt.vlines(x[len(x)-1], 0, 0.1, linestyle = ':')
plt.text(x[len(x)-1], 0.1, '2022-03-23', fontsize=20)
heights = [0.15, 0.15, 0.10, 0.2, 0.17, 0.13, 0.13, 0.13]
vtype= ['Alpha\nBeta\nGamma', 'Epsil\nIota\nZeta', 'Kappa',
        'Theta', 'Lambda', 'Delta', 'Mu', 'Omicron']
for i,rw in enumerate(retraining_week):
    week_date_list = list(set(metadata[metadata['week'] == rw][col_submission_date].tolist()))
    week_date = str(min([get_time(x) for x in week_date_list]))
    plt.vlines(rw-2, 0, heights[i],
               #transform=ax.get_xaxis_transform(),
               linestyle='-', color='red')
    plt.text(rw-2, heights[i], week_date+'\n'+vtype[i], rotation=0, fontsize=20)

# plt.xticks(x, labels=[str(y+2) for y in x], rotation=45, fontsize=25)

plt.xticks(x, labels=newlab_x, rotation=0, fontsize=28)

plt.yticks(fontsize=28)
plt.xlabel('Week', fontsize=25)
plt.title('Percentage of predicted anomalies from 2020-07-06 to 2022-03-23', fontsize=25)
plt.savefig('/mnt/resources/2022_04/2022_04/percentage_anomalies_v2.png', dpi=350)




""" Performance """
first_time = {19:'Alpha', }

x = np.arange(len(count_nonneutral))
w=0.35

fig, ax = plt.subplots(figsize=(20,10))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.rcParams.update({'font.size': 12})
plt.bar([y-w*0.5 for y in x], predicted_neural, label='Predicted Neutral', width=w, alpha=0.8)
plt.bar([y+w*0.5 for y in x], predicted_nonneutral, label='Predicted anomaly', width=w, alpha=0.8)
plt.vlines(x[0], 0, 8000, linestyle = ':')
plt.text(x[0], 8000, '2020-07-06')
plt.vlines(x[len(x)-1], 0, 8000, linestyle=':')
plt.text(x[len(x)-1], 8000, '2022-03-23')
heights = [9000, 9000, 9000, 10000, 9500, 9000, 9000, 9000]
vtype= ['Alpha\nBeta\nGamma', 'Epsil\nIota\nZeta', 'Kappa',
        'Theta', 'Lambda', 'Delta', 'Mu', 'Omicron']
for i,rw in enumerate(retraining_week):
    week_date_list = list(set(metadata[metadata['week'] == rw][col_submission_date].tolist()))
    week_date = str(min([get_time(x) for x in week_date_list]))
    plt.vlines(rw-2, 0, heights[i],
               #transform=ax.get_xaxis_transform(),
               linestyle='-', color='red')
    plt.text(rw-2, heights[i], week_date+'\n'+vtype[i], rotation=0)


plt.vlines(19-2, 0, 2000,
               #transform=ax.get_xaxis_transform(),
               linestyle='-', color='red')
plt.scatter(19-2, 2010,
               #transform=ax.get_xaxis_transform(),
               linestyle='-', color='red',
            marker='s', s=60)
plt.text(19-3, 2000, '2020-11-02\nAlpha', rotation=0)

plt.xticks(x, labels=[str(y+2) for y in x], rotation=45)
plt.xlabel('Week')
plt.legend()
plt.show()


label_class = ['', 'Alpha', 'Beta', 'Gamma', 'Epsil', 'Iota', 'Zeta','Kappa',
               'Theta', 'Lambda', 'Delta', 'Mu', 'Omicron', '']
y = np.arange(len(label_class))
fig, ax = plt.subplots(figsize=(32,14))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.rcParams.update({'font.size': 12})

# when we find the Alpha
plt.vlines(19-2, 0, y[1], color='red')
plt.scatter(19-2, y[1], s=70, marker='x',
            linewidths=10, color='red', label='Predicted as anomaly')
# when the CDC declared alpha as VOC
plt.vlines(27-2, 0, y[1], color='blue')
plt.scatter(27-2, y[1], s=70, marker='o', linewidths=10, color='blue',
            label='Recognized as VOC/VOI')
plt.hlines(y[1], 0, 27-2, linestyles='--')

# when we find the Beta
plt.vlines(20-2, 0, y[2], color='red')
plt.scatter(20-2, y[2], s=70, marker='x', linewidths=10, color='red')
# when the CDC declared beta as VOC
plt.vlines(27-2, 0, y[2], color='blue')
plt.scatter(27-2, y[2], s=70, marker='o', linewidths=10, color='blue')
plt.hlines(y[2], 0, 27-2, linestyles='--')

# when we find the Gamma
plt.vlines(27-2, 0, y[3], color='red')
plt.scatter(27-2, y[3], s=70, marker='x',
            alpha=0.2,linewidths=10, color='red')
# when the CDC declared gamma as VOC
plt.vlines(27-2, 0, y[3], color='blue')
plt.scatter(27-2, y[3], s=70,
            alpha=0.5, marker='o', linewidths=10, color='blue')
plt.hlines(y[3], 0, 27-2, linestyles='--')

# when we find the Epsil
plt.vlines(22-2, 0, y[4], color='red')
plt.scatter(22-2, y[4], s=70, marker='x', linewidths=10, color='red')
# when the CDC declared epsil as VOC
plt.vlines(35-2, 0, y[4], color='blue')
plt.scatter(35-2, y[4], s=70, marker='o',
            linewidths=10, color='blue')
plt.hlines(y[4],0, 35-2, linestyles='--')

# when we find the Zeta
plt.vlines(23-2, 0, y[6], color='red')
plt.scatter(23-2, y[6], s=70, marker='x', linewidths=10, color='red')
# when the CDC declared zeta as VOC
plt.vlines(35-2, 0, y[6], color='blue')
plt.scatter(35-2, y[6], s=70, marker='o', linewidths=10, color='blue')
plt.hlines(y[6],0, 35-2, linestyles='--')

# when we find the kappa
plt.vlines(32-2, 0, y[7], color='red')
plt.scatter(32-2, y[7], s=70, marker='x', linewidths=10, color='red')
# when the CDC declared kappa as VOC
plt.vlines(45-2, 0, y[7], color='blue')
plt.scatter(45-2, y[7], s=70, marker='o', linewidths=10, color='blue')
plt.hlines(y[7],0, 45-2, linestyles='--')



# when we find the theta
plt.vlines(40-2, 0, y[8], color='red')
plt.scatter(40-2, y[8], s=70, marker='x', linewidths=10, color='red')
# when the CDC declared theta as VOC
plt.vlines(48-2, 0, y[8], color='blue')
plt.scatter(48-2, y[8], s=70, marker='o', linewidths=10, color='blue')
plt.hlines(y[8],0, 48-2, linestyles='--')


# when we find the iota
plt.vlines(29-2, 0, y[5], color='red')
plt.scatter(29-2, y[5], s=70, marker='x', linewidths=10, color='red')
# when the CDC declared iota as Voc
plt.vlines(35-2, 0, y[5], color='blue')
plt.scatter(35-2, y[5], s=70, marker='o', linewidths=10, color='blue')
plt.hlines(y[5],0, 35-2, linestyles='--')

# when we find the lambda
plt.vlines(29-2, 0, y[9], color='red')
plt.scatter(29-2, y[9], s=70, marker='x', linewidths=10, color='red')
# when the CDC declared lambda as voc
plt.vlines(49-2, 0, y[9], color='blue')
plt.scatter(49-2, y[9], s=70, marker='o', linewidths=10, color='blue')
plt.hlines(y[9],0, 49-2, linestyles='--')


# when we find the delta
plt.vlines(31-2, 0, y[10], color='red')
plt.scatter(31-2, y[10], s=70, marker='x', linewidths=10, color='red')
# when the CDC declared delta as voc
plt.vlines(51-2, 0, y[10], color='blue')
plt.scatter(51-2, y[10], s=70, marker='o', linewidths=10, color='blue')
plt.hlines(y[10],0, 51-2, linestyles='--')

# when we find the mu
plt.vlines(41-2, 0, y[11], color='red')
plt.scatter(41-2, y[11], s=70, marker='x', linewidths=10, color='red')
# when the CDC declared mu as voc
plt.vlines(62-2, 0, y[11], color='blue')
plt.scatter(62-2, y[11], s=70, marker='o', linewidths=10, color='blue')
plt.hlines(y[11],0, 62-2, linestyles='--')

# when we find the omicron
plt.vlines(72-2, 0, y[12], color='red')
plt.scatter(72-2, y[12], s=70, marker='x', linewidths=10, color='red')
# when the CDC declared mu as voc
plt.vlines(75-2, 0, y[12], color='blue')
plt.scatter(75-2, y[12], s=70, marker='o', linewidths=10, color='blue')
plt.hlines(y[12],0, 75-2, linestyles='--')


#plt.xticks(x[0:81], labels=[str(k+2) for k in x[0:81]], rotation=45, fontsize=20)
plt.xticks(x, newlab_x, fontsize=28)
plt.yticks(y, labels=label_class, rotation=0, fontsize=25)
plt.xlabel('Week', fontsize=25)
plt.legend(fontsize=25)
plt.title('Detection of VOC/VOI as anomaly and comparison\nwith date of designation according to CDC',
          fontsize=25)
plt.savefig('/mnt/resources/2022_04/2022_04/performance_oneclass_v2.png', dpi=350)

# y_test_predicted_class
for variant in metadata[col_class_variant].unique():
    nvar = []
    npred_anomal = []
    for k in results_fine_tune[0]['y_test_variant_type'].keys():
        true_class_list = results_fine_tune[0]['y_test_variant_type'][k]
        pred_class_list = results_fine_tune[0]['y_test_predicted_class'][k]
        i_v = np.where(np.array(true_class_list)==variant)[0]
        if len(i_v) != 0:
            i_p = np.where(np.array(pred_class_list)[i_v]==-1)[0]
            npred_anomal.append(len(i_p))
        else:
            npred_anomal.append(0)
        nvar.append(len(i_v))

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.rcParams.update({'font.size': 12})
    plt.bar(x, nvar, label=variant, width=w, alpha=0.8)
    plt.bar(x,  npred_anomal, bottom=nvar, label=variant+' predicted anomaly', width=w, alpha=0.8)


    plt.xticks(x, labels=[str(y + 2) for y in x], rotation=45)
    plt.xlabel('Week')
    plt.legend()
    plt.title(variant+' distibution in GISAID')

    plt.savefig('/mnt/resources/2022_04/2022_04/distribution_'+variant+'.png', dpi=150)
