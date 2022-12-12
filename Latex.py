import matplotlib.pyplot as plt
import pylatex as pl
import os
import json
import argparse
from pylatex import Document, Section, Subsection, Tabular, MultiColumn,\
    MultiRow, Command, Table
from pylatex.utils import NoEscape


def get_directories(root):
    return [folder for folder in os.listdir(root)
            if os.path.isdir(os.path.join(root, folder))]

def get_json_file(root):

    files = [folder for folder in os.listdir(root)
            if not os.path.isdir(os.path.join(root, folder))]

    for file in files:
        if file.endswith('json'):
            return os.path.join(root, file)

    return None

def convert_to_latex_table(results, doc_name):
    doc = Document(doc_name)
    table = Table()
    total_columns = 1
    tabular = Tabular('|' +'c|'*9)
    tabular.add_hline()
    tabular.add_row(('Model Name', 'Dataset', 'JSD', 'RMS', 'Count', 'Mean', 'Stdev', 'FPS', NoEscape('$\Delta$count')))
    tabular.add_hline()

    for result in results:
        tabular.add_hline()
        for i in range(4):
            if(i>0):
                tabular.add_hline(2, )
            dataset_name = str(i)
            row_cells = []
            if(i==0):
                row_cells.append(MultiRow(4, data=' '.join(result['exp_name'].split('_'))))
            else:
                row_cells.append('')
            row_cells += [dataset_name, round(result['results'][dataset_name]['jsd2_50_ft'], 3), round(result['results'][dataset_name]['RMS'], 3), round(result['results'][dataset_name]['seed_count_pred_50ft'], 3), round(result['results'][dataset_name]['Mean_Predicted'], 3), round(result['results'][dataset_name]['stdev_predicted'], 3), round(result['results'][dataset_name]['fps'],3), result['results'][dataset_name]['seed_count_pred_50ft'] - result['results'][dataset_name]['seeed_count_valid_50ft']]
            tabular.add_row(row_cells)
        tabular.add_hline()


    table.append(tabular)
    table.add_caption('Results')
    doc.append(table)
    doc.generate_pdf(clean_tex=False)

def group_all_results(results_list):
    ret = {'0' : {}, '1' : {}, '2' : {}, '3': {}}
    for result in results_list:
        for i in range(4):
            dataset_name = str(i)
            for key in result['results'][dataset_name].keys():
                print(key, dataset_name, result['results'][dataset_name][key])
                if(not key in ret[dataset_name].keys()):
                    ret[dataset_name][key] = []
                ret[dataset_name][key].append(result['results'][dataset_name][key])
    print(ret)
    return ret


def box_plot(results_dataset, metric):
    data = []
    for key in results_dataset.keys():
        print(metric, results_dataset[key][metric])
        data.append(results_dataset[key][metric])
    print(data)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.boxplot(data)
    ax.set_xticklabels(range(4))
    ax.set_xlabel('Dataset Name')
    return ax

if __name__ == "__main__":
    training_outputs_dir = "/home/r4hul/dataset/06-30-evaled/Training_Outputs/"

    exps = [folder for folder in os.listdir(training_outputs_dir)
            if os.path.isdir(os.path.join(training_outputs_dir, folder))]

    list_dirs = []

    for exp in exps:
        ret_dict = {}
        ret_dict['exp_name'] = exp
        ret_dict['exp_path'] = os.path.join(training_outputs_dir, exp, 'SLS')
        ret_dict['results_path'] = get_json_file(ret_dict['exp_path'])
        with open(ret_dict['results_path'], 'r') as f:
            ret_dict['results'] = json.load(f)
        list_dirs.append(ret_dict)

    convert_to_latex_table(results=list_dirs, doc_name='Test')
    results_dataset = group_all_results(list_dirs)

    datapoints = u'Δ count'
    for dataset in results_dataset.keys():
        results_dataset[dataset][datapoints] = [x -y for x, y in zip(results_dataset[dataset]['seed_count_pred_50ft'],results_dataset[dataset]['seeed_count_valid_50ft'])]

    mean_label = u'ΔMean'
    for dataset in results_dataset.keys():
        results_dataset[dataset][mean_label] = [x -y for x, y in zip(results_dataset[dataset]['Mean_Predicted'],results_dataset[dataset]['Mean_Valid'])]


    y_labels = ['JSD',          'JSD',           'Seed Count (N)',       'Seed Count (N)',     'JSD',          'JSD',          'RMS (m)',      'Mean (m)',      'Mean (m)',      'Stdev (m)',      'Stdev (m)',      'FPS (N)', u'ΔCount (N)', u'ΔMean (m)']
    titles   = ['JSD Box Plot', 'JSD Box Plot', 'Seed Count Box Plot',  'Seed Count Box Plot', 'JSD Box Plot', 'JSD Box Plot', 'RMS Box Plot', 'Mean Box Plot', 'Mean Box Plot', 'Stdev Box Plot', 'Stdev Box Plot', 'FPS Box Plot', u'ΔCount Box Plot', u'ΔMean Box Plot']
    for i, metric in enumerate(results_dataset['0'].keys()):
        ax = box_plot(results_dataset, metric)
        ax.set_ylabel(y_labels[i])
        ax.set_title(titles[i])
        plt.rc('font', size=12)
        plt.savefig(metric+'.JPG')

