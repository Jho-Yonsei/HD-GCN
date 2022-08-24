import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm
import math

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)
    parser.add_argument('--alpha1',
                        default=0.1,
                        type=float)
    parser.add_argument('--main-dir',
                        help='')
    parser.add_argument('--CoM-2',
                        type=str2bool,
                        default=True)
    parser.add_argument('--CoM-21',
                        type=str2bool,
                        default=True)
    parser.add_argument('--CoM-1',
                        type=str2bool,
                        default=True)
    
    arg = parser.parse_args()

    dataset = arg.dataset
    if 'UCLA' in arg.dataset:
        label = []
        with open('./data/' + 'NW-UCLA/' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CS.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError
    
    dir_cnt = 0

    if arg.CoM_1:
        with open(os.path.join(arg.main_dir, 'joint_CoM_1/', 'epoch1_test_score.pkl'), 'rb') as r1:
            r1 = list(pickle.load(r1).items())
        with open(os.path.join(arg.main_dir, 'bone_CoM_1/', 'epoch1_test_score.pkl'), 'rb') as r2:
            r2 = list(pickle.load(r2).items())
        dir_cnt += 2

    if arg.CoM_2:
        with open(os.path.join(arg.main_dir, 'joint_CoM_2/', 'epoch1_test_score.pkl'), 'rb') as r3:
            r3 = list(pickle.load(r3).items())
        with open(os.path.join(arg.main_dir, 'bone_CoM_2/', 'epoch1_test_score.pkl'), 'rb') as r4:
            r4 = list(pickle.load(r4).items())
        dir_cnt += 2
        
    if arg.CoM_21:
        if 'ntu' in arg.dataset:
            with open(os.path.join(arg.main_dir, 'joint_CoM_21/' 'epoch1_test_score.pkl'), 'rb') as r5:
                r5 = list(pickle.load(r5).items())
            with open(os.path.join(arg.main_dir, 'bone_CoM_21/', 'epoch1_test_score.pkl'), 'rb') as r6:
                r6 = list(pickle.load(r6).items())
            dir_cnt += 2
        elif 'UCLA' in arg.dataset:
            with open(os.path.join(arg.main_dir, 'joint_CoM_3/' 'epoch1_test_score.pkl'), 'rb') as r5:
                r5 = list(pickle.load(r5).items())
            with open(os.path.join(arg.main_dir, 'bone_CoM_3/', 'epoch1_test_score.pkl'), 'rb') as r6:
                r6 = list(pickle.load(r6).items())
            dir_cnt += 2

    right_num = total_num = right_num_5 = 0
    
    def norm(x):
        x_norm = x / x.std()
        return x_norm

    if dir_cnt == 6:
        for i in tqdm(range(len(label))):
            l = label[i]
            r11 = np.array(r1[i][1])
            r22 = np.array(r2[i][1])
            r33 = np.array(r3[i][1])
            r44 = np.array(r4[i][1])
            r55 = np.array(r5[i][1])
            r66 = np.array(r6[i][1])
            r11 = norm(r11)
            r22 = norm(r22)
            r33 = norm(r33)
            r44 = norm(r44)
            r55 = norm(r55)
            r66 = norm(r66)
            r = r11 + r22 + r33 + r44 + r55 + r66
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
        
    elif dir_cnt == 4:
        r = None
        for i in tqdm(range(len(label))):
            l = label[i]
            if arg.CoM_1:
                r11 = np.array(r1[i][1])
                r22 = np.array(r2[i][1])
                r = norm(r11) + norm(r22)
            if arg.CoM_2:
                r33 = np.array(r3[i][1])
                r44 = np.array(r4[i][1])
                r = r + norm(r33) + norm(r44) if r is not None else norm(r33) + norm(r44)
            if arg.CoM_21:
                r55 = np.array(r5[i][1])
                r66 = np.array(r6[i][1])
                r = r + norm(r55) + norm(r66) if r is not None else norm(r55) + norm(r66)

            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num

    elif dir_cnt == 2:
        r = None
        for i in tqdm(range(len(label))):
            l = label[i]
            if arg.CoM_1:
                r11 = np.array(r1[i][1])
                r22 = np.array(r2[i][1])
                r = norm(r11) + norm(r22)       
            if arg.CoM_2:
                r33 = np.array(r3[i][1])
                r44 = np.array(r4[i][1])
                r = r + norm(r33) + norm(r44) if r is not None else norm(r33) + norm(r44)
            if arg.CoM_21:
                r55 = np.array(r5[i][1])
                r66 = np.array(r6[i][1])
                r = r + norm(r55) + norm(r66) if r is not None else norm(r55) + norm(r66)

            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
    print()
    print('In Paper: {:.1f}%'.format(acc * 100))
