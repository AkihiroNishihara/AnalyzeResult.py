# coding: utf-8
"""
結果格納クラス，分析クラス作成
1. 引数で獲得したメソッドの出力を獲得
2.

dict_label["DS"]["method"][trial]["tra"/"tst"]["true"/generation] -> [label]
dict_ruleset["DS"]["method"][trial][i]="000240010a00"
"""
# package(standard)
import header as h
import os
import sys
import itertools
from collections import defaultdict
import pprint
import json
import statistics
# package(third party)
import sklearn.metrics as metrics
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# TODO:ベースメソッドの列を有意差のファイルでは無視する
dict_class_major = {}

# list_method = ["original", "replica", "SMOTE", "IDFGBML"]

# aquired in load_true_and_pred
list_dirname_method = []

LIST_DS = []
LIST_METHOD = []
TRIAL = 20
GENERATION = 5000
DIV_GRAPH = 100

# 評価指標の個数
NUM_EVALUATION = 1


# ["DS"]["method"][trial]["tra"/"tst"]["true"/generation] -> [label]
def load_dict_label(_path_dir_output):
    dict_label = {}

    list_dir = os.listdir(_path_dir_output)
    for dirname_method in list_dir:
        list_dirname_method.append(dirname_method)
        method = dirname_method.replace("IDFGBML_", "")
        method = method.replace("FGBML_", "")
        LIST_METHOD.append(method)
        for ds in LIST_DS:
            if ds not in dict_label.keys():
                dict_label[ds] = {}
            path_dir_output_ds = _path_dir_output + "/{0}/result/{1}".format(dirname_method, ds)
            if os.path.exists(path_dir_output_ds) is False:
                continue
            list_filename_result = os.listdir(path_dir_output_ds)

            # read file
            count_file = 0
            dict_label_file = {}
            for filename_result in list_filename_result:
                fp = open(path_dir_output_ds + "/" + filename_result, 'r')
                list_line = fp.readlines()

                # read line
                index_start_pred = list_line.index("<Prediction> [generation][index_pattern]\n") + 1
                list_line_pred = list_line[index_start_pred:]
                index_mid = int(len(list_line_pred) / 2)
                list_line_pred_tra = list_line_pred[1:int(GENERATION / DIV_GRAPH + 3)]
                list_line_pred_tst = list_line_pred[index_mid + 1:index_mid + 1 + int(GENERATION / DIV_GRAPH + 2)]
                dict_pred_tra = {}
                for i in range(len(list_line_pred_tra)):
                    list_str = list_line_pred_tra[i].replace('\n', '').split(',')
                    list_label = [int(string) for string in list_str[1:]]
                    if i == 0:
                        dict_pred_tra.update({'true': list_label})
                    else:
                        dict_pred_tra.update({int(list_str[0]): list_label})
                dict_pred_tst = {}
                for i in range(len(list_line_pred_tst)):
                    list_str = list_line_pred_tst[i].replace('\n', '').split(',')
                    list_label = [int(string) for string in list_str[1:]]
                    if i == 0:
                        dict_pred_tst.update({'true': list_label})
                    else:
                        dict_pred_tst.update({int(list_str[0]): list_label})
                dict_label_file.update({count_file: {'tra': dict_pred_tra, 'tst': dict_pred_tst}})
                count_file += 1

            dict_label[ds].update({method: dict_label_file})

    return dict_label


def load_ruleset(_path_dir_output):
    dict_ruleset = {}

    list_dir = os.listdir(_path_dir_output)
    for dirname_method in list_dir:
        list_dirname_method.append(dirname_method)
        method = dirname_method.replace("IDFGBML_", "")
        method = method.replace("FGBML_", "")
        for ds in LIST_DS:
            if ds not in dict_ruleset.keys():
                dict_ruleset[ds] = {}
            path_dir_output_ds = _path_dir_output + "/{0}/result/{1}".format(dirname_method, ds)
            if os.path.exists(path_dir_output_ds) is False:
                continue
            list_filename_result = os.listdir(path_dir_output_ds)

            # read file
            count_file = 0
            dict_rule_file = {}
            for filename_result in list_filename_result:
                fp = open(path_dir_output_ds + "/" + filename_result, 'r')
                list_line = fp.readlines()

                # read line
                index_start_ruleset = list_line.index("<BestRules> [att, class, weight]\n") + 1
                list_rule = []
                for line in list_line[index_start_ruleset:]:
                    if line is '\n':
                        break
                    list_rule.append(line.split(',')[0])
                dict_rule_file[count_file] = list_rule
                count_file += 1

            dict_ruleset[ds].update({method: dict_rule_file})

    return dict_ruleset


# ["DS"]["method"][trial]["tra"/"tst"][generation] -> GM
def get_dict_GM_allTrial_allGene(_dict_label_pred):
    __dict_GM_allTrial_allGene = {}

    # CMの獲得
    for ds in LIST_DS:
        dict_GM_ds = {}
        num_class = max(_dict_label_pred[ds][LIST_METHOD[0]][0]['tra']['true']) + 1
        for method in LIST_METHOD:
            dict_GM_ds_method = {}
            for trial in range(TRIAL):
                dict_label_tra = _dict_label_pred[ds][method][trial]['tra']
                dict_trial_tra = {gene: get_GM(dict_label_tra['true'], dict_label_tra[gene], num_class) for gene in
                                  range(0, GENERATION + 1, DIV_GRAPH)}
                dict_label_tst = _dict_label_pred[ds][method][trial]['tst']
                dict_trial_tst = {gene: get_GM(dict_label_tst['true'], dict_label_tst[gene], num_class) for gene in
                                  range(0, GENERATION + 1, DIV_GRAPH)}
                dict_GM_ds_method[trial] = {'tra': dict_trial_tra, 'tst': dict_trial_tst}
            dict_GM_ds[method] = dict_GM_ds_method
        __dict_GM_allTrial_allGene[ds] = dict_GM_ds

    return __dict_GM_allTrial_allGene


def get_GM(_list_label_true, _list_label_pred, _num_class):
    # CM = metrics.confusion_matrix(y_true=_list_label_true, y_pred=_list_label_pred,
    #                               labels=[i for i in range(-1, _num_class)])
    # if -1 in _list_label_pred:
    #     pprint.pprint(CM)
    list_recall = metrics.recall_score(_list_label_true, _list_label_pred, average=None)
    GM = stats.mstats.gmean(list_recall[-1 * _num_class:])
    return GM


# dict_GM_last['DS']['method']['tra'/'tst'][trial] = GM(last_gene)
def get_dict_GM_last(_dict_GM_allTrial_allGene):
    dict_last_GM = {}
    for ds in LIST_DS:
        dict_GM_ds = {}
        for method in LIST_METHOD:
            dict_GM_ds_method = {'tra': {}, 'tst': {}}
            dict_GM_temp_method = _dict_GM_allTrial_allGene[ds][method]
            for trial in range(TRIAL):
                dict_GM_ds_method['tra'][trial] = dict_GM_temp_method[trial]['tra'][GENERATION]
                dict_GM_ds_method['tst'][trial] = dict_GM_temp_method[trial]['tst'][GENERATION]
            dict_GM_ds[method] = dict_GM_ds_method
        dict_last_GM[ds] = dict_GM_ds

    return dict_last_GM


# dict_graph_GM['DS']['method']['tra'/'tst'][gene] = GM(average_trial)
def get_dict_graph_GM(_dict_GM_allTrial_allGene):
    dict_graph_GM = {}
    for ds in LIST_DS:
        dict_GM_ds = {}
        for method in LIST_METHOD:
            dict_GM_ds_method = {'tra': {gene: 0.0 for gene in range(0, GENERATION + 1, DIV_GRAPH)},
                                 'tst': {gene: 0.0 for gene in range(0, GENERATION + 1, DIV_GRAPH)}}
            for trial in range(TRIAL):
                dict_GM_tra_allGene = _dict_GM_allTrial_allGene[ds][method][trial]['tra']
                dict_GM_tst_allGene = _dict_GM_allTrial_allGene[ds][method][trial]['tst']
                for gene in range(0, GENERATION + 1, DIV_GRAPH):
                    dict_GM_ds_method['tra'][gene] += dict_GM_tra_allGene[gene] / TRIAL
                    dict_GM_ds_method['tst'][gene] += dict_GM_tst_allGene[gene] / TRIAL
            dict_GM_ds[method] = dict_GM_ds_method
        dict_graph_GM[ds] = dict_GM_ds

    return dict_graph_GM


def write_line_pointers(_list_line, _list_pointers):
    for line, pt in zip(_list_line, _list_pointers):
        pt.write(line)


def check_wilcoxon(_list_comp, _list_base):
    ans = 0  # -1: worse with significant difference, 0: without sig-dif, 1: better with sig-dif
    # print("{0}, {1}\n".format(len(_list_comp), len(_list_base)))
    if _list_comp == _list_base:
        ans = 0
    else:
        test_wilcoxon = stats.wilcoxon(np.array(_list_comp), np.array(_list_base), zero_method='pratt')
        if test_wilcoxon.pvalue < 0.05:
            if sum(_list_comp) < sum(_list_base):
                ans = 1
            else:
                ans = -1
    return ans


# 外部から読み込み
def load_dict_class_major():
    fp = open("namelist_dataset_major.txt", 'r')
    for line in fp.readlines():
        nameDS_major = (line.replace('\n', '')).split(',')
        dict_class_major[nameDS_major[0]] = int(nameDS_major[1])


def get_list_ds_sorted_IR():
    fp = open("namelist_dataset_major.txt", 'r')
    global LIST_DS
    LIST_DS = [(name_ds.split(','))[0] for name_ds in fp.readlines()]


def plot_graph(_dict_graph_GM):
    plt.rcParams['font.family'] = 'Times New Roman'  # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix'  # math fontの設定
    plt.rcParams["font.size"] = 20  # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 20  # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 20  # 軸だけ変更されます
    list_color = ["green", "blue", "red"]
    os.makedirs("./graph", exist_ok=True)
    for ds in LIST_DS:
        list_x = []
        count_method = 0
        # plt.rcParams['figure.figsize'] = (3.5, 3.5)  # figure size in inch, 横×縦
        plt.rcParams['figure.dpi'] = 300
        # fig = plt.figure()
        # plt.title(ds)
        fig, ax = plt.subplots(1, 1)
        plt.xlabel("Generation")
        plt.ylabel("Average GM")
        plt.xticks(np.arange(0, 5001, 1000))
        plt.ylim([0.0, 1.0])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        for method in LIST_METHOD:
            dict_graph_tra = _dict_graph_GM[ds][method]['tra']
            dict_graph_tst = _dict_graph_GM[ds][method]['tst']
            if len(list_x) == 0:
                list_x = [int(key) for key in dict_graph_tra.keys()]
            list_y_tra = list(dict_graph_tra.values())
            list_y_tst = list(dict_graph_tst.values())
            plt.plot(list_x, list_y_tra, color=list_color[count_method], label=method + "(tra)", linestyle="--")
            plt.plot(list_x, list_y_tst, color=list_color[count_method], label=method + "(tst)")
            # plt.legend()
            # plt.show()
            count_method += 1
        # fig.savefig("./graph/graph_{0}.eps".format(ds), bbox_inches="tight", pad_inches=0.05)
        fig.savefig("./graph/graph_{0}.png".format(ds), bbox_inches="tight", pad_inches=0.05)
        plt.clf()
        plt.close()


def main():
    args = sys.argv
    num_support_pattern = str(args[1])
    method_base = args[2]
    # load_dict_class_major()
    get_list_ds_sorted_IR()

    path_dir_output = "./output(H={0}_dist=n_class=a)".format(num_support_pattern)
    dict_time = defaultdict(list)  # {DS:[method][trial]}
    """
    # Step.1 load label and calculate evaluation
    print("load label-data: start\n")
    dict_label_pred = load_dict_label(path_dir_output)
    print("load label-data: end\n")

    print("calculate GM_allTrial_allGene: start\n")
    dict_GM_allTrial_allGene = get_dict_GM_allTrial_allGene(dict_label_pred)
    print("calculate GM_allTrial_allGene: end\n")

    # dict_GM_last['DS']['method']['tra'/'tst'][trial] = GM(last_gene)
    # dict_graph_GM['DS']['method']['tra'/'tst'][gene] = GM(average_trial)
    dict_GM_last = get_dict_GM_last(dict_GM_allTrial_allGene)
    dict_graph_GM = get_dict_graph_GM(dict_GM_allTrial_allGene)
    os.makedirs("./json_files", exist_ok=True)
    with open("./json_files/dict_GM_last.json", 'w') as fp:
        json.dump(dict_GM_last, fp)
    with open("./json_files/dict_graph_GM.json", 'w') as fp:
        json.dump(dict_graph_GM, fp)
    """

    # Step.2 load json and evaluate
    dict_GM_last = json.load(open('./json_files/dict_GM_last.json', 'r'))
    dict_graph_GM = json.load(open('./json_files/dict_graph_GM.json', 'r'))
    # dict_ruleset["DS"]["method"][trial][i] = "000240010a00"
    dict_ruleset = load_ruleset(path_dir_output)
    LIST_METHOD.append('NONE')
    LIST_METHOD.append('SMOTE')
    LIST_METHOD.append('FTFT')
    plot_graph(dict_graph_GM)

    # make unity_output
    path_dir_unity_output = path_dir_output.replace("./output", "./unity_output")
    os.makedirs(path_dir_unity_output, exist_ok=True)

    fp_gmean = open(path_dir_unity_output + "/unity_gmean.csv", 'w')
    fp_sd_gmean = open(path_dir_unity_output + "/unity_stdev_gmean.csv", 'w')
    fp_dif_gmean = open(path_dir_unity_output + "/unity_dif_gmean.csv", 'w')

    # output 1st line
    str_1st_line = ""
    for method in LIST_METHOD:
        str_1st_line += "," + method
    str_1st_line += '\n'
    list_str_1st = [str_1st_line] * NUM_EVALUATION
    write_line_pointers(list_str_1st, [fp_gmean])
    write_line_pointers(list_str_1st, [fp_sd_gmean])
    write_line_pointers(list_str_1st, [fp_dif_gmean])

    for ds in LIST_DS:
        str_gmean = ds
        str_sd_gmean = ds
        dict_list_gmean = defaultdict(list)
        for method in LIST_METHOD:
            list_gmean = [value for value in dict_GM_last[ds][method]['tst'].values()]
            str_gmean += "," + str(np.average(list_gmean))
            str_sd_gmean += "," + str(statistics.stdev(list_gmean))
            dict_list_gmean[method] = list_gmean

        fp_gmean.write(str_gmean + "\n")
        fp_sd_gmean.write(str_sd_gmean + "\n")

        # output each line
        fp_dif_gmean.write(ds + ',')

        for method_comp in LIST_METHOD:
            if method_comp == method_base:
                fp_dif_gmean.write("=,")
                continue

            dif_gmean = check_wilcoxon(dict_list_gmean[method_comp], dict_list_gmean[method_base])
            if dif_gmean == -1:
                fp_dif_gmean.write("-,")
            elif dif_gmean == 1:
                fp_dif_gmean.write("+,")
            else:
                fp_dif_gmean.write("/,")

        fp_dif_gmean.write("\n")

    # evaluate num_rule
    fp_ruleset = open(path_dir_unity_output + "/unity_num_rule.csv", 'w')
    fp_sd_ruleset = open(path_dir_unity_output + "/unity_stdev_num_rule.csv", 'w')
    fp_dif_ruleset = open(path_dir_unity_output + "/unity_dif_num_rule.csv", 'w')

    # output 1st line
    str_1st_line = ""
    for method in LIST_METHOD:
        str_1st_line += "," + method
    str_1st_line += '\n'
    list_str_1st = [str_1st_line] * NUM_EVALUATION
    write_line_pointers(list_str_1st, [fp_ruleset])
    write_line_pointers(list_str_1st, [fp_sd_ruleset])
    write_line_pointers(list_str_1st, [fp_dif_ruleset])

    for ds in LIST_DS:
        str_ruleset = ds
        str_sd_ruleset = ds
        dict_list_ruleset = defaultdict(list)
        for method in LIST_METHOD:
            # dict_ruleset["DS"]["method"][trial][i]="000240010a00"
            list_num_ruleset = [len(dict_ruleset[ds][method][trial]) for trial in range(TRIAL)]
            str_ruleset += "," + str(np.average(list_num_ruleset))
            str_sd_ruleset += "," + str(statistics.stdev(list_num_ruleset))
            dict_list_ruleset[method] = list_num_ruleset

        fp_ruleset.write(str_ruleset + "\n")
        fp_sd_ruleset.write(str_sd_ruleset + "\n")

        # output each line
        fp_dif_ruleset.write(ds + ',')

        for method_comp in LIST_METHOD:
            if method_comp == method_base:
                fp_dif_ruleset.write("=,")
                continue

            dif_ruleset = check_wilcoxon(dict_list_ruleset[method_comp], dict_list_ruleset[method_base])
            if dif_ruleset == -1:
                fp_dif_ruleset.write("-,")
            elif dif_ruleset == 1:
                fp_dif_ruleset.write("+,")
            else:
                fp_dif_ruleset.write("/,")

        fp_dif_ruleset.write("\n")


    # get dict of graph_data (ds, method, trial) = {gene: AUC}
    # dict_graph_data = get_graph_data()

    # # make unity_graph: filename = "unity_graph_DATASET.dat", [gene][method] = value
    # path_dir_unity_graph = path_dir_output.replace("./output", "./unity_graph")
    # os.makedirs(path_dir_unity_graph, exist_ok=True)
    # fp_graph = open(path_dir_unity_graph + "/unity_graph_AUC_{0}.csv".format(ds), 'w')
    # fp_graph.write(str_1st_line+"\n")
    #


if __name__ == '__main__':
    main()
