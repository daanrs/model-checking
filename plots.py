import json
import matplotlib.pyplot as plt
import numpy as np

def load_data(file='data.json'):
    with open(file, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    df = load_data()
    # print(df)


    for name, alg in df.items():
        fig, ax = plt.subplots()

        ax.set_xscale('log')
        
        for alg_name, results in alg.items():
            x = np.array([r[0] for r in results])
            y = np.array([r[1] for r in results])

            ax.plot(x, y, label=alg_name)

        ax.legend()

        plt.savefig(f'plots/{name}.pdf')
            
