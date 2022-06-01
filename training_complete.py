from main import *
from datetime import date
import matplotlib.pyplot as plt
import random

class MakeGraph:

    def __init__(self, txt_file):
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
        plt.legend()
        return fig

    def parse_model(self, lines):
        model = {}
        model['name'] = lines[0].split(' ')[-2]
        model['IOUS'] = list(map(lambda x: float(x), lines[1].split()[2:]))
        model['P'] = list(map(lambda x: float(x), lines[2].split()[2:]))
        model['R'] = list(map(lambda x: float(x), lines[3].split()[2:]))
        model['AP'] = list(map(lambda x: float(x), lines[4].split()[2:]))
        return model

    def show_fig(self):
        self.make_graph()
        plt.show()

    def save_fig(self, fig_name):
        self.make_graph
        plt.savefig(fig_name)
        plt.show()

if __name__ == "__main__":
    content = "Trained YOLORP6, YOLORCSPX, YOLOVCSPX"
    fig_name = 'PvsR.jpeg'
    txt_file = f'/home/r4hul/{date.today().strftime("%m-%d")}_ap.txt'
    gh = MakeGraph(txt_file=txt_file)
    gh.save_fig(fig_name)
    send_mail(["r4hul@ksu.edu"], content, "Training Complete", txt_file)