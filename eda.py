import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import REPORTS


def main():
    data = pd.read_csv(REPORTS / 'dataset_summary.csv').set_index('class')
    for columns, name in [(['raw','unique'], 'class_distribution'), (['train','test'], 'split_distribution')]:
        ax = data[columns].plot.bar(rot=0, figsize=(8,5), ylabel='Images')
        ax.figure.tight_layout()
        ax.figure.savefig(REPORTS / f'{name}.png', dpi=150)
        plt.close(ax.figure)
    data[['train','test']].to_csv(REPORTS / 'split_distribution.csv')
    print(data)

if __name__ == '__main__':
    main()
