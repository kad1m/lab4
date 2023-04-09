# # x1_dim -  розмірність X1
# # x2_dim - розмірність X2
# # x3_dim - розмірність X3
# # y_dim - розмірність Y
# # pred_steps - кроки прогнозування
# # pred_samples - розмір вибірки для прогнозування
# from sklearn.ensemble import IsolationForest
# from read_file import ReadFile
# from ui.backend.utils import UtilsCalculation
# import numpy as np
#
#
# x1_dim = 4
# x2_dim = 2
# x3_dim = 3
# y_dim = 3
#
# x1 = 0
# x2 = 0
# x3 = 0
#
# pred_samples = 50
# pred_steps = 10
#
# danger_levels = [
#     (11.7, 10.5),
#     (1, 0),
#     (11.7, 10.5)
# ]
#
#
# # rf = ReadFile('C:\ROG\pythonProjects\lab4_cs\data.txt')
# # data = np.array(rf.get_data()).reshape(-1, 1+sum([x1_dim, x2_dim, x3_dim, y_dim]))
# path = 'C:\ROG\pythonProjects\lab4_cs\data.txt'
# uc = UtilsCalculation()
#
#
# def get_fault_probs(input_data, y_dim):
#     fault_probs = []
#     for i in range(y_dim):
#         fault_probs.append(
#             uc.moving_average_prediction(
#                 input_data[:, -y_dim + i],
#                 y_emergency=danger_levels[i][0],
#                 y_fatal=danger_levels[i][1],
#                 window_size=pred_samples // pred_steps
#             )
#         )
#     return np.array(fault_probs).T
#
#
# def get_isolation_forest_predict(x):
#     all_pred = []
#     for i in range(x.shape[1]):
#         clf = IsolationForest(
#             random_state=42,
#         ).fit(x[:, i].reshape(-1, 1))
#         pred = clf.predict(x[:, i].reshape(-1, 1))
#         all_pred.append(pred)
#     all_pred = np.array(all_pred)
#     return 1 * (all_pred.sum(axis=0) == -2)
#
#
# with open(path, 'r') as file:
#     input_data = np.fromstring('\n'.join(file.read().split('\n')[1:]), sep='\t').reshape(
#         -1, 1+sum([x1_dim, x2_dim, x3_dim, y_dim])
#     )
#
#
# fault_probs = get_fault_probs(input_data, y_dim)
# rdr = ['0.00%'] * (pred_samples - 1)
#
# predict = get_isolation_forest_predict(input_data[:, 1:x1_dim+1])
# print(predict)
#
#
