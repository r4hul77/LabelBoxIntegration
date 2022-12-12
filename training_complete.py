from datetime import date
import matplotlib.pyplot as plt
import random
import os
import json

class MakeGraph:

    def __init__(self, txt_file, json_file):
        self.txt_file = txt_file


    def make_graph(self):
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
        fig = plt.figure()
        fig.suptitle('Precession Vs Recall Graphs')
        number_models = len(lines)//5
        models = []
        for i in range(number_models):
            models.append(self.parse_model(lines[i*5:(i+1)*5]))

        for model in models:
            plt.plot(model['R'], model['P'], label=model['name'])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(0, 1.5)
        plt.ylim(0, 1.5)
        plt.legend()
        return fig

    def parse_model(self, lines):
        model = {}
        model['name'] = lines[0].split(' ')[-2]
        model['IOUS'] = list(map(lambda x: float(x), lines[1].split()[2:]))
        model['P'] = list(map(lambda x: float(x), lines[2].split()[2:]))
        model['R'] = list(map(lambda x: float(x), lines[3].split()[2:]))
        #model['AP'] = list(map(lambda x: float(x), lines[4].split()[2:]))
        return model

    def show_fig(self):
        self.make_graph()
        plt.show()

    def save_fig(self, fig_name):
        self.make_graph()
        plt.savefig(fig_name)
        plt.show()
        plt.close()

class makeJsonGraph:

    def __init__(self):
        pass

    def make_graph(self, json_file):
        fig = plt.figure()
        fig.suptitle('Precession Vs Recall Graphs')
        with open(json_file, 'r') as f:
            obj = json.load(f)
        for model in obj['models']:
            plt.plot(model['recall'], model['precision'], label=' '.join([model['name'], "{:03f}".format(model['Ap'])]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(0, 1.5)
        plt.ylim(0, 1.5)
        plt.legend()
        return fig

    def save_fig(self, json_file, save_path):

        fig = self.make_graph(json_file)
        plt.savefig(save_path)
        plt.show(block=False)
        plt.pause(3)
        plt.close()

if __name__ == "__main__":
    content = "Trained YOLOR-P6, YOLOR-CSPX, YOLOV4-CSPX, YOLOX-S, YOLOX-M, YOLOX-L"
    fig_name = 'PvsR.jpeg'
    training_dir = '/home/r4hul/dataset/07-11/Training_Outputs'
    txt_file = os.path.join(training_dir, 'ap_doubles.txt')
    save_path = os.path.join(training_dir, 'PRcurve_doubles.jpeg')
    gh = makeJsonGraph()
    gh.save_fig(txt_file, save_path)
    #send_mail(["r4hul@ksu.edu"], content, "Training Complete", txt_file)
