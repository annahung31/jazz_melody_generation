import midi
import glob
import random
import numpy as np
import pretty_midi
import seaborn as sns
sns.set()
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from mgeval_src import core, utils
from sklearn.model_selection import LeaveOneOut 


set1 = glob.glob('/nas2/ai_music_database/jazz_freejammaster/split/*.mid')
# random.seed(2)
random.seed(1)
num_samples = 100
set1 = random.sample(set1, num_samples)

num_sets = 16
model_list = [0] * num_sets
model_list[0] = set1
model_list[1] = glob.glob('/nas2/annahung/project/anna_jam_v2/paper_outputs_jazz_only/no_pcd_bata_0.7_en_layer_2/jamming/loss*.mid')
model_list[2] = glob.glob('/nas2/annahung/project/anna_jam_v2/paper_outputs_jazz_only/bata_0.7_en_layer_2/jamming/loss*.mid')
model_list[3] = glob.glob('/nas2/annahung/project/anna_jam_v2/paper_outputs_adaptation_no_pcd/bata_0.7_en_layer_2_ratio_6.0/jamming/loss_tt*.mid')
model_list[4] = glob.glob('/nas2/annahung/project/anna_jam_v2/paper_outputs_conditional_one_hot/bata_0.7_en_layer_2_ratio_1.0/jamming_one_hot/loss*.mid')
model_list[5] = glob.glob('/nas2/annahung/project/anna_jam_v2/paper_outputs_conditional_one_hot/bata_0.7_en_layer_2_ratio_2.0/jamming_one_hot/loss*.mid')
model_list[6] = glob.glob('/nas2/annahung/project/anna_jam_v2/paper_outputs_conditional_one_hot/bata_0.7_en_layer_2_ratio_3.0/jamming_one_hot/loss*.mid')
model_list[7] = glob.glob('/nas2/annahung/project/anna_jam_v2/paper_outputs_conditional_one_hot/bata_0.7_en_layer_2_ratio_4.0/jamming_one_hot/loss*.mid')
model_list[8] = glob.glob('/nas2/annahung/project/anna_jam_v2/paper_outputs_conditional_one_hot/bata_0.7_en_layer_2_ratio_5.0/jamming_one_hot/loss*.mid')
model_list[9] = glob.glob('/nas2/annahung/project/anna_jam_v2/paper_outputs_conditional_one_hot/bata_0.7_en_layer_2_ratio_6.0/jamming_one_hot/loss*.mid')
model_list[10] = glob.glob('/nas2/annahung/project/anna_jam_v2/paper_outputs_adaptation_no_pcd/bata_0.7_en_layer_2_ratio_1.0/jamming/lossno*.mid')
model_list[11] = glob.glob('/nas2/annahung/project/anna_jam_v2/paper_outputs_adaptation_no_pcd/bata_0.7_en_layer_2_ratio_2.0/jamming/lossno*.mid')
model_list[12] = glob.glob('/nas2/annahung/project/anna_jam_v2/paper_outputs_adaptation_no_pcd/bata_0.7_en_layer_2_ratio_3.0/jamming/lossno*.mid')
model_list[13] = glob.glob('/nas2/annahung/project/anna_jam_v2/paper_outputs_adaptation_no_pcd/bata_0.7_en_layer_2_ratio_4.0/jamming/lossno*.mid')
model_list[14] = glob.glob('/nas2/annahung/project/anna_jam_v2/paper_outputs_adaptation_no_pcd/bata_0.7_en_layer_2_ratio_5.0/jamming/lossno*.mid')
model_list[15] = glob.glob('/nas2/annahung/project/anna_jam_v2/paper_outputs_adaptation_no_pcd/bata_0.7_en_layer_2_ratio_6.0/jamming/lossno*.mid')


assert len(model_list[1]) == num_samples
assert len(model_list[2]) == num_samples
assert len(model_list[3]) == num_samples
assert len(model_list[4]) == num_samples
assert len(model_list[5]) == num_samples
assert len(model_list[6]) == num_samples
assert len(model_list[7]) == num_samples
assert len(model_list[8]) == num_samples
assert len(model_list[9]) == num_samples
assert len(model_list[10]) == num_samples
assert len(model_list[11]) == num_samples
assert len(model_list[12]) == num_samples
assert len(model_list[13]) == num_samples
assert len(model_list[14]) == num_samples
assert len(model_list[15]) == num_samples



dir_name = 'result/3_'

# print('evaluating total_used_note')
# num_bar = 4
# set1_eval = {'total_used_note':np.zeros((num_samples,1))}
# metrics_list = set1_eval.keys()
# for i in range(0, num_samples):
#     feature = core.extract_feature(set1[i])
#     set1_eval[metrics_list[0]][i] = getattr(core.metrics(), metrics_list[0])(feature, track_num= 1)

# for i in range(4,10):
#     set2 = model_list[i]
#     set2_eval = {'total_used_note':np.zeros((num_samples,1))}
#     for i in range(0, num_samples):
#         feature = core.extract_feature(set2[i])
#         set2_eval[metrics_list[0]][i] = getattr(core.metrics(), metrics_list[0])(feature, track_num= 1)


#     loo = LeaveOneOut()
#     loo.get_n_splits(np.arange(num_samples))
#     set1_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
#     set2_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
#     for i in range(len(metrics_list)):
#         for train_index, test_index in loo.split(np.arange(num_samples)):
#             now_matrix = metrics_list[i]
#             set1_intra[test_index[0]][i] = utils.c_dist(set1_eval[now_matrix][test_index], set1_eval[now_matrix][train_index])
#             set2_intra[test_index[0]][i] = utils.c_dist(set2_eval[now_matrix][test_index], set2_eval[now_matrix][train_index])

#     loo = LeaveOneOut()
#     loo.get_n_splits(np.arange(num_samples))
#     sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))
#     for i in range(len(metrics_list)):
#         for train_index, test_index in loo.split(np.arange(num_samples)):
#             sets_inter[test_index[0]][i] = utils.c_dist(set1_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]])


#     plot_set1_intra = np.transpose(set1_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
#     plot_set2_intra = np.transpose(set2_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
#     plot_sets_inter = np.transpose(sets_inter,(1, 0, 2)).reshape(len(metrics_list), -1)

#     file_name = dir_name +'1total_used_note'+'.txt'
#     with open(file_name, "a") as text_file:
#         for i in range(0, len(metrics_list)):
#             # print metrics_list[i] + ':'
#             # print '------------------------'
#             # print ' demo_set1'
#             OA = utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
#             print OA
#             text_file.write(str(OA))
#             text_file.write('\n')


# print('evaluating bar_used_note')
# num_bar = 4
# set1_eval = {'bar_used_note':np.zeros((num_samples,num_bar,1))}
# metrics_list = set1_eval.keys()
# for i in range(0, num_samples):
#     feature = core.extract_feature(set1[i])
#     set1_eval[metrics_list[0]][i] = getattr(core.metrics(), metrics_list[0])(feature, track_num= 1 , num_bar = num_bar)

# for i in range(4,10):
#     set2 = model_list[i]
#     set2_eval = {'bar_used_note':np.zeros((num_samples,num_bar,1))}
#     for i in range(0, num_samples):
#         feature = core.extract_feature(set2[i])
#         set2_eval[metrics_list[0]][i] = getattr(core.metrics(), metrics_list[0])(feature, track_num= 1 , num_bar = num_bar)


#     loo = LeaveOneOut()
#     loo.get_n_splits(np.arange(num_samples))
#     set1_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
#     set2_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
#     for i in range(len(metrics_list)):
#         for train_index, test_index in loo.split(np.arange(num_samples)):
#             now_matrix = metrics_list[i]
#             set1_intra[test_index[0]][i] = utils.c_dist(set1_eval[now_matrix][test_index], set1_eval[now_matrix][train_index])
#             set2_intra[test_index[0]][i] = utils.c_dist(set2_eval[now_matrix][test_index], set2_eval[now_matrix][train_index])

#     loo = LeaveOneOut()
#     loo.get_n_splits(np.arange(num_samples))
#     sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))
#     for i in range(len(metrics_list)):
#         for train_index, test_index in loo.split(np.arange(num_samples)):
#             sets_inter[test_index[0]][i] = utils.c_dist(set1_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]])



#     plot_set1_intra = np.transpose(set1_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
#     plot_set2_intra = np.transpose(set2_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
#     plot_sets_inter = np.transpose(sets_inter,(1, 0, 2)).reshape(len(metrics_list), -1)

#     file_name = dir_name +'2bar_used_note'+'.txt'
#     with open(file_name, "a") as text_file:
#         for i in range(0, len(metrics_list)):
#             # print metrics_list[i] + ':'
#             # print '------------------------'
#             # print ' demo_set1'
#             OA = utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
#             print OA
#             text_file.write(str(OA))
#             text_file.write('\n')




# print('evaluating note_length_hist')
# num_bar = 4
# set1_eval = {'note_length_hist':np.zeros((num_samples,12))}
# metrics_list = set1_eval.keys()
# for i in range(0, num_samples):
#     feature = core.extract_feature(set1[i])
#     set1_eval[metrics_list[0]][i] = getattr(core.metrics(), metrics_list[0])(feature, track_num=1, normalize=True, pause_event=False)

# for i in range(4,10):
#     set2 = model_list[i]
#     set2_eval = {'note_length_hist':np.zeros((num_samples,12))}
#     for i in range(0, num_samples):
#         feature = core.extract_feature(set2[i])
#         set2_eval[metrics_list[0]][i] = getattr(core.metrics(), metrics_list[0])(feature, track_num=1, normalize=True, pause_event=False)


#     loo = LeaveOneOut()
#     loo.get_n_splits(np.arange(num_samples))
#     set1_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
#     set2_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
#     for i in range(len(metrics_list)):
#         for train_index, test_index in loo.split(np.arange(num_samples)):
#             now_matrix = metrics_list[i]
#             set1_intra[test_index[0]][i] = utils.c_dist(set1_eval[now_matrix][test_index], set1_eval[now_matrix][train_index])
#             set2_intra[test_index[0]][i] = utils.c_dist(set2_eval[now_matrix][test_index], set2_eval[now_matrix][train_index])

#     loo = LeaveOneOut()
#     loo.get_n_splits(np.arange(num_samples))
#     sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))
#     for i in range(len(metrics_list)):
#         for train_index, test_index in loo.split(np.arange(num_samples)):
#             sets_inter[test_index[0]][i] = utils.c_dist(set1_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]])



#     plot_set1_intra = np.transpose(set1_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
#     plot_set2_intra = np.transpose(set2_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
#     plot_sets_inter = np.transpose(sets_inter,(1, 0, 2)).reshape(len(metrics_list), -1)

#     file_name = dir_name +'3note_length_hist'+'.txt'
#     with open(file_name, "a") as text_file:
#         for i in range(0, len(metrics_list)):
#             # print metrics_list[i] + ':'
#             # print '------------------------'
#             # print ' demo_set1'
#             OA = utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
#             print OA
#             text_file.write(str(OA))
#             text_file.write('\n')


# print('evaluating note_length_transition_matrix')
# num_bar = 4
# set1_eval = {'note_length_transition_matrix':np.zeros((num_samples, 12, 12))}
# metrics_list = set1_eval.keys()
# for i in range(0, num_samples):
#     feature = core.extract_feature(set1[i])
#     set1_eval[metrics_list[0]][i] = getattr(core.metrics(), metrics_list[0])(feature, track_num=1, normalize=0, pause_event=False)

# for i in range(4,10):
#     set2 = model_list[i]
#     set2_eval = {'note_length_transition_matrix':np.zeros((num_samples, 12, 12))}
#     for i in range(0, num_samples):
#         feature = core.extract_feature(set2[i])
#         set2_eval[metrics_list[0]][i] = getattr(core.metrics(), metrics_list[0])(feature, track_num=1, normalize=0, pause_event=False)

#     loo = LeaveOneOut()
#     loo.get_n_splits(np.arange(num_samples))
#     set1_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
#     set2_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
#     for i in range(len(metrics_list)):
#         for train_index, test_index in loo.split(np.arange(num_samples)):
#             now_matrix = metrics_list[i]
#             set1_intra[test_index[0]][i] = utils.c_dist(set1_eval[now_matrix][test_index], set1_eval[now_matrix][train_index])
#             set2_intra[test_index[0]][i] = utils.c_dist(set2_eval[now_matrix][test_index], set2_eval[now_matrix][train_index])

#     loo = LeaveOneOut()
#     loo.get_n_splits(np.arange(num_samples))
#     sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))
#     for i in range(len(metrics_list)):
#         for train_index, test_index in loo.split(np.arange(num_samples)):
#             sets_inter[test_index[0]][i] = utils.c_dist(set1_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]])



#     plot_set1_intra = np.transpose(set1_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
#     plot_set2_intra = np.transpose(set2_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
#     plot_sets_inter = np.transpose(sets_inter,(1, 0, 2)).reshape(len(metrics_list), -1)

#     file_name = dir_name +'4note_length_transition_matrix'+'.txt'
#     with open(file_name, "a") as text_file:
#         for i in range(0, len(metrics_list)):
#             # print metrics_list[i] + ':'
#             # print '------------------------'
#             # print ' demo_set1'
#             OA = utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
#             print OA
#             text_file.write(str(OA))
#             text_file.write('\n')



# print('evaluating total_used_pitch')
# num_bar = 4
# set1_eval = {'total_used_pitch':np.zeros((num_samples,1))}
# metrics_list = set1_eval.keys()
# for i in range(0, num_samples):
#     feature = core.extract_feature(set1[i])
#     set1_eval[metrics_list[0]][i] = getattr(core.metrics(), metrics_list[0])(feature)

# for i in range(4,10):
#     set2 = model_list[i]
#     set2_eval = {'total_used_pitch':np.zeros((num_samples,1))}
#     for i in range(0, num_samples):
#         feature = core.extract_feature(set2[i])
#         set2_eval[metrics_list[0]][i] = getattr(core.metrics(), metrics_list[0])(feature)


#     loo = LeaveOneOut()
#     loo.get_n_splits(np.arange(num_samples))
#     set1_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
#     set2_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
#     for i in range(len(metrics_list)):
#         for train_index, test_index in loo.split(np.arange(num_samples)):
#             now_matrix = metrics_list[i]
#             set1_intra[test_index[0]][i] = utils.c_dist(set1_eval[now_matrix][test_index], set1_eval[now_matrix][train_index])
#             set2_intra[test_index[0]][i] = utils.c_dist(set2_eval[now_matrix][test_index], set2_eval[now_matrix][train_index])

#     loo = LeaveOneOut()
#     loo.get_n_splits(np.arange(num_samples))
#     sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))
#     for i in range(len(metrics_list)):
#         for train_index, test_index in loo.split(np.arange(num_samples)):
#             sets_inter[test_index[0]][i] = utils.c_dist(set1_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]])

#     plot_set1_intra = np.transpose(set1_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
#     plot_set2_intra = np.transpose(set2_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
#     plot_sets_inter = np.transpose(sets_inter,(1, 0, 2)).reshape(len(metrics_list), -1)

#     file_name = dir_name +'5total_used_pitch'+'.txt'
#     with open(file_name, "a") as text_file:
#         for i in range(0, len(metrics_list)):
#             # print metrics_list[i] + ':'
#             # print '------------------------'
#             # print ' demo_set1'
#             OA = utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
#             print OA
#             text_file.write(str(OA))
#             text_file.write('\n')



# print('evaluating bar_used_pitch')
# num_bar = 4
# set1_eval = {'bar_used_pitch':np.zeros((num_samples,num_bar,1))}
# metrics_list = set1_eval.keys()
# for i in range(0, num_samples):
#     feature = core.extract_feature(set1[i])
#     set1_eval[metrics_list[0]][i] = getattr(core.metrics(), metrics_list[0])(feature, track_num= 1 , num_bar = num_bar)

# for i in range(4,10):
#     set2 = model_list[i]
#     set2_eval = {'bar_used_pitch':np.zeros((num_samples,num_bar,1))}
#     for i in range(0, num_samples):
#         feature = core.extract_feature(set2[i])
#         set2_eval[metrics_list[0]][i] = getattr(core.metrics(), metrics_list[0])(feature, track_num= 1 , num_bar = num_bar)


#     loo = LeaveOneOut()
#     loo.get_n_splits(np.arange(num_samples))
#     set1_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
#     set2_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
#     for i in range(len(metrics_list)):
#         for train_index, test_index in loo.split(np.arange(num_samples)):
#             now_matrix = metrics_list[i]
#             set1_intra[test_index[0]][i] = utils.c_dist(set1_eval[now_matrix][test_index], set1_eval[now_matrix][train_index])
#             set2_intra[test_index[0]][i] = utils.c_dist(set2_eval[now_matrix][test_index], set2_eval[now_matrix][train_index])

#     loo = LeaveOneOut()
#     loo.get_n_splits(np.arange(num_samples))
#     sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))
#     for i in range(len(metrics_list)):
#         for train_index, test_index in loo.split(np.arange(num_samples)):
#             sets_inter[test_index[0]][i] = utils.c_dist(set1_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]])



#     plot_set1_intra = np.transpose(set1_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
#     plot_set2_intra = np.transpose(set2_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
#     plot_sets_inter = np.transpose(sets_inter,(1, 0, 2)).reshape(len(metrics_list), -1)

#     file_name = dir_name +'6bar_used_pitch'+'.txt'
#     with open(file_name, "a") as text_file:
#         for i in range(0, len(metrics_list)):
#             # print metrics_list[i] + ':'
#             # print '------------------------'
#             # print ' demo_set1'
#             OA = utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
#             print OA
#             text_file.write(str(OA))
#             text_file.write('\n')



# print('evaluating pitch_range')
# num_bar = 4
# set1_eval = {'pitch_range':np.zeros((num_samples,1))}
# metrics_list = set1_eval.keys()
# for i in range(0, num_samples):
#     feature = core.extract_feature(set1[i])
#     set1_eval[metrics_list[0]][i] = getattr(core.metrics(), metrics_list[0])(feature)

# for i in range(4,10):
#     set2 = model_list[i]
#     set2_eval = {'pitch_range':np.zeros((num_samples,1))}
#     for i in range(0, num_samples):
#         feature = core.extract_feature(set2[i])
#         set2_eval[metrics_list[0]][i] = getattr(core.metrics(), metrics_list[0])(feature)

#     loo = LeaveOneOut()
#     loo.get_n_splits(np.arange(num_samples))
#     set1_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
#     set2_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
#     for i in range(len(metrics_list)):
#         for train_index, test_index in loo.split(np.arange(num_samples)):
#             now_matrix = metrics_list[i]
#             set1_intra[test_index[0]][i] = utils.c_dist(set1_eval[now_matrix][test_index], set1_eval[now_matrix][train_index])
#             set2_intra[test_index[0]][i] = utils.c_dist(set2_eval[now_matrix][test_index], set2_eval[now_matrix][train_index])

#     loo = LeaveOneOut()
#     loo.get_n_splits(np.arange(num_samples))
#     sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))
#     for i in range(len(metrics_list)):
#         for train_index, test_index in loo.split(np.arange(num_samples)):
#             sets_inter[test_index[0]][i] = utils.c_dist(set1_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]])

#     plot_set1_intra = np.transpose(set1_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
#     plot_set2_intra = np.transpose(set2_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
#     plot_sets_inter = np.transpose(sets_inter,(1, 0, 2)).reshape(len(metrics_list), -1)

#     file_name = dir_name +'7pitch_range'+'.txt'
#     with open(file_name, "a") as text_file:
#         for i in range(0, len(metrics_list)):
#             # print metrics_list[i] + ':'
#             # print '------------------------'
#             # print ' demo_set1'
#             OA = utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
#             print OA
#             text_file.write(str(OA))
#             text_file.write('\n')



# print('evaluating total_pitch_class_histogram')
# num_bar = 4
# set1_eval = {'total_pitch_class_histogram':np.zeros((num_samples, 12))}
# metrics_list = set1_eval.keys()
# for i in range(0, num_samples):
#     feature = core.extract_feature(set1[i])
#     set1_eval[metrics_list[0]][i] = getattr(core.metrics(), metrics_list[0])(feature)

# for i in range(4,10):
#     set2 = model_list[i]
#     set2_eval = {'total_pitch_class_histogram':np.zeros((num_samples, 12))}
#     for i in range(0, num_samples):
#         feature = core.extract_feature(set2[i])
#         set2_eval[metrics_list[0]][i] = getattr(core.metrics(), metrics_list[0])(feature)


#     loo = LeaveOneOut()
#     loo.get_n_splits(np.arange(num_samples))
#     set1_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
#     set2_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
#     for i in range(len(metrics_list)):
#         for train_index, test_index in loo.split(np.arange(num_samples)):
#             now_matrix = metrics_list[i]
#             set1_intra[test_index[0]][i] = utils.c_dist(set1_eval[now_matrix][test_index], set1_eval[now_matrix][train_index])
#             set2_intra[test_index[0]][i] = utils.c_dist(set2_eval[now_matrix][test_index], set2_eval[now_matrix][train_index])

#     loo = LeaveOneOut()
#     loo.get_n_splits(np.arange(num_samples))
#     sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))
#     for i in range(len(metrics_list)):
#         for train_index, test_index in loo.split(np.arange(num_samples)):
#             sets_inter[test_index[0]][i] = utils.c_dist(set1_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]])



#     plot_set1_intra = np.transpose(set1_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
#     plot_set2_intra = np.transpose(set2_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
#     plot_sets_inter = np.transpose(sets_inter,(1, 0, 2)).reshape(len(metrics_list), -1)

#     file_name = dir_name +'8total_pitch_class_histogram'+'.txt'
#     with open(file_name, "a") as text_file:
#         for i in range(0, len(metrics_list)):
#             # print metrics_list[i] + ':'
#             # print '------------------------'
#             # print ' demo_set1'
#             OA = utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
#             print OA
#             text_file.write(str(OA))
#             text_file.write('\n')





#total_used_note
print('evaluating pitch_class_transition_matrix')
num_bar = 4
set1_eval = {'pitch_class_transition_matrix':np.zeros((num_samples, 12, 12))}
metrics_list = set1_eval.keys()
for i in range(0, num_samples):
    feature = core.extract_feature(set1[i])
    set1_eval[metrics_list[0]][i] = getattr(core.metrics(), metrics_list[0])(feature, normalize=0)


PCTM = np.nanmean(set1_eval['pitch_class_transition_matrix'],axis=0)
ax = sns.heatmap(PCTM, vmin=0, vmax=1, xticklabels=['C', 'C#', 'D', 'D#','E',  'F',  'F#', 'G', 'Ab','A', 'Bb','B'],yticklabels=['C', 'C#', 'D', 'D#','E',  'F',  'F#', 'G', 'G#','A', 'Bb','B'])
fig = ax.get_figure()
fig.savefig("training_data_pctm.pdf")
plt.close()




for i in range(1,16):
    set2 = model_list[i]
    set2_eval = {'pitch_class_transition_matrix':np.zeros((num_samples, 12, 12))}
    for j in range(0, num_samples):
        feature = core.extract_feature(set2[j])
        set2_eval[metrics_list[0]][j] = getattr(core.metrics(), metrics_list[0])(feature, normalize=0)


    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    set1_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
    set2_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
    for k in range(len(metrics_list)):
        for train_index, test_index in loo.split(np.arange(num_samples)):
            now_matrix = metrics_list[k]
            set1_intra[test_index[0]][k] = utils.c_dist(set1_eval[now_matrix][test_index], set1_eval[now_matrix][train_index])
            set2_intra[test_index[0]][k] = utils.c_dist(set2_eval[now_matrix][test_index], set2_eval[now_matrix][train_index])

    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))
    for l in range(len(metrics_list)):
        for train_index, test_index in loo.split(np.arange(num_samples)):
            sets_inter[test_index[0]][l] = utils.c_dist(set1_eval[metrics_list[l]][test_index], set2_eval[metrics_list[l]])



    plot_set1_intra = np.transpose(set1_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
    plot_set2_intra = np.transpose(set2_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
    plot_sets_inter = np.transpose(sets_inter,(1, 0, 2)).reshape(len(metrics_list), -1)

    file_name = dir_name +'9pitch_class_transition_matrix'+'.txt'
    with open(file_name, "a") as text_file:
        for m in range(0, len(metrics_list)):
            # print metrics_list[i] + ':'
            # print '------------------------'
            # print ' demo_set1'
            OA = utils.overlap_area(plot_set1_intra[m], plot_sets_inter[m])
            print OA
            text_file.write(str(OA))
            text_file.write('\n')
    
    PCTM = np.nanmean(set2_eval['pitch_class_transition_matrix'],axis=0)
    ax = sns.heatmap(PCTM, vmin=0, vmax=1, xticklabels=['C', 'C#', 'D', 'Eb','E',  'F',  'F#', 'G', 'Ab','A', 'Bb','B'],yticklabels=['C', 'C#', 'D', 'Eb','E',  'F',  'F#', 'G', 'Ab','A', 'Bb','B'])
    fig = ax.get_figure()
    fig.savefig('model_'+str(i) + "pctm.pdf")
    plt.close()




