import pandas as pd
from sklearn.metrics import f1_score

# csv_file = "SELab-HCMUS_Submission/pbdang_submission.csv"
# csv_file = "SELab-HCMUS_Submission/final_result.csv"
csv_file = "SELab-HCMUS_Submission/khoi_submission.csv"
# csv_file = "SELab-HCMUS_Submission/tan_ngoan_result.csv"
# csv_file = "ensemble.csv"

label_file = "mediaeval-visual-sentiment-test/testset.csv"

all_labels_1 = {'negative':0,'neutral': 1,'positive':2}


label = pd.read_csv(label_file)
label.sort_values(["filename"], axis=0, ascending=[False], inplace=True)

pred = pd.read_csv(csv_file)
pred.sort_values(["id"], axis=0, ascending=[False], inplace=True)


#############################
########## TASK 1 ###########
#############################
y_pred = []
y_true = []

for i in range(len(pred)):
    y_pred.append(all_labels_1[pred.iloc[i]['T1']])
    y_true.append(all_labels_1[label.iloc[i]['T1']])

f1_w_1 = f1_score(y_true, y_pred, average='weighted')
f1_u_1 = f1_score(y_true, y_pred, average='micro')
f1_m_1 = f1_score(y_true, y_pred, average='macro')
print("Task 1:")
print("F1-score weighted: ", f1_w_1)
print("F1-score micro: ", f1_u_1)
print("F1-score macro: ", f1_m_1)



#############################
########## TASK 2 ###########
#############################
y_pred = []
y_true = []

for i in range(len(pred)):
    # T2.1: Joy,T2.2: Sadness,T2.3: Fear,T2.4: Disgust,T2.5: Anger,T2.6: Surprise,T2.7: Neutral
    tmp_pred = pred.iloc[i][2:9].tolist()
    tmp_label = label.iloc[i][3:10].tolist()

    y_pred.append(tmp_pred)
    y_true.append(tmp_label)


f1_w_2 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
f1_u_2 = f1_score(y_true, y_pred, average='micro', zero_division=0)
f1_m_2 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("Task 2:")
print("F1-score weighted: ", f1_w_2)
print("F1-score micro: ", f1_u_2)
print("F1-score macro: ", f1_m_2)



#############################
########## TASK 3 ###########
#############################
y_pred = []
y_true = []

for i in range(len(pred)):
    tmp_pred = pred.iloc[i][9:19].tolist()
    tmp_label = label.iloc[i][10:20].tolist()

    y_pred.append(tmp_pred)
    y_true.append(tmp_label)

f1_w_3 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
f1_u_3 = f1_score(y_true, y_pred, average='micro', zero_division=0)
f1_m_3 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("Task 3:")
print("F1-score weighted: ", f1_w_3)
print("F1-score micro: ", f1_u_3)
print("F1-score macro: ", f1_m_3)


