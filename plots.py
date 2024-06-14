import json
import matplotlib.pyplot as plt
import numpy as np

def load_data(file='data.json'):
    with open(file, 'r') as f:
        return json.load(f)


titles = {
    'bet_fav': 'Betting Game Favourable',
    'bet_unfav': 'Betting Game Unfavourable',
    'bandit': 'Bandit'
}

if __name__ == '__main__':
    df = load_data()

    for name, alg in df.items():
        fig, ax = plt.subplots()

        ax.set_xscale('log')
        ax.set_xlabel('Trajectory')
        ax.set_ylabel('R [F T]')
        
        for alg_name, results in alg.items():
            x = np.array([r[0] for r in results])
            y = np.array([r[1] for r in results])

            ax.plot(x, y, label=alg_name)

        ax.legend()
        ax.set_title(titles[name])

        plt.savefig(f'plots/{name}.pdf')
            
