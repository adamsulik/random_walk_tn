import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def random_generator(n, gamma, epsilon):
    """"Generates n random numbers from f(x) ~ x^gamma, truncated to x = <epsilon; 1>"""
    def inverse_dist(F):
        """returns alpha"""
        eps_powered = pow(epsilon, 1-gamma)
        return pow(F*(1-eps_powered)+eps_powered, 1/(1-gamma))
    return inverse_dist(np.random.random(n))


def base_func(alpha, gamma, epsilon):
    c = (1-gamma)/(1 - pow(epsilon, 1-gamma))
    return c * np.power(alpha, -gamma)


class TemporalNet:
    """
    Class of temporal network. It consists of fields:\n
    n - number of agents in the network \n
    m - number of connections on each activation \n
    walkers - list of indices where are walkers, i-th indices corresponds to the i-th walker spot \n
    connection_matrix - matrix of connections symmetrical about the diagonal \n
    a_vec - vector of activation values of each agent; i-th value corresponds to the i-th agent
    """

    def __init__(self, n: int, m: int, walkers_num: int, random_a_func, funcargs):
        self.n = n
        self.m = m
        self.walkers = np.random.randint(0, n, walkers_num).tolist()
        self.connection_matrix = np.zeros((n, n))
        self.a_vec = random_a_func(n, *funcargs)

    def step(self):
        self.connection_matrix.fill(0)

        # activate agents with specified probability
        self.activated = np.where(np.random.random(self.n) < self.a_vec)[0]

        # assign connections
        for active in self.activated:
            available_agents = np.append(np.arange(0, active), np.arange(active+1, self.n))
            chosen_agents = np.random.choice(available_agents, self.m, replace=False)
            for chosen in chosen_agents:
                self.connection_matrix[(active, chosen), (chosen, active)] = 1

        # walk
        walksX, walksY = np.where(self.connection_matrix > 0)
        new_walkers = []
        for walker in self.walkers:
            # move walker to other node if possible
            walks = np.append(walksY[np.where(walksX == walker)], walksX[np.where(walksY == walker)])
            new_walkers.append(np.random.choice(walks) if walks.size else walker)
        self.walkers = new_walkers

    def get_walkers_hist(self, start, end, bins: int = 30):
        bins = np.logspace(np.log10(start), np.log10(end), bins+1)
        walkers_a = self.a_vec[self.walkers]
        hist, bins = np.histogram(walkers_a, bins=bins)
        mids = np.power(10, (np.log10(bins[1:]) + np.log10(bins[:-1]))/2)
        return hist, mids


def save_plot(out_path, a_vec, gamma, epsilon):
    fig, ax = plt.subplots(figsize=(10, 10))
    x = np.linspace(epsilon, 1, 5000)
    y = base_func(x, gamma, epsilon)
    bins = 30
    bins = np.logspace(np.log10(epsilon), np.log10(1), bins + 1)
    hist, bins = np.histogram(a_vec, bins=bins, density=True)
    mids = np.power(10, (np.log10(bins[1:]) + np.log10(bins[:-1])) / 2)
    ax.plot(x, y, label=r'distribution function of $\mathcal{a}$ values')
    ax.scatter(mids, hist, label='histogram of example $\mathcal{a}$ values')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('activation')
    ax.set_ylabel('density')
    ax.legend()
    fig.savefig(out_path, format='jpg')


def main(args):
    # Prepare output
    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.save_plot:
        figs_dir = save_dir / 'figs'
        figs_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'output.csv', 'a') as f:
        f.write(f'#n={args.agents_num}\n'
                f'#m={args.m_value}\n'
                f'#repetitions={args.repetitions}\n'
                f'#walkers fraction={args.walkers_fraction}\n'
                f'#steps={args.steps}\n'
                f'#gamma={args.gamma}\n'
                f'#epsilon={args.eps}\n')
        f.write('sim_id;n;gamma;a;Wa\n')

    # run simulations
    sim_id = 0
    for n in args.agents_num:
        for gamma in args.gamma:
            for rep in range(args.repetitions):
                tnet = TemporalNet(n=n, m=args.m_value, walkers_num=int(args.walkers_fraction*n),
                                   random_a_func=random_generator, funcargs=(gamma, args.eps))
                if rep == 0 and args.save_plot:
                    save_plot(figs_dir / f'n{n}_gamma{gamma}.jpg', tnet.a_vec, gamma, args.eps)
                for _ in tqdm(range(args.steps)):
                    tnet.step()
                hist, a_mids = tnet.get_walkers_hist(args.eps, 1)
                with open(save_dir / 'output.csv', 'a') as f:
                    for c, a in zip(hist, a_mids):
                        f.write(f"{sim_id};{n};{gamma};{a};{c}\n")
                sim_id += 1

if __name__ == "__main__":
    now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--repetitions', default=1, type=int,
                        help='Repetitions of each simulation, possibly to calculate average results')
    parser.add_argument('-n', '--agents-num', type=int, nargs='*', default=[100])
    parser.add_argument('-m', '--m-value', type=int, default=2)
    parser.add_argument('-s', '--steps', type=int, default=30)
    parser.add_argument('-w', '--walkers-fraction', type=float, default=0.1, help='Fraction of walkers and agents '
                                                                                  'number, based on that number of '
                                                                                  'walkers will be calculated')
    parser.add_argument('-g', '--gamma', type=float, nargs='*', default=[2.], help='Distribution of activation values '
                                                                                   'is f(x)~x^{-gamma}; choose gamma')
    parser.add_argument('-e', '--eps', type=float, default=0.1, help='Activation values are between epsilon and 1')
    parser.add_argument('-o', '--output', type=Path, default=Path(f'outputs/{now}'), help='Path to save outputs')
    parser.add_argument('--save-plot', action='store_true', help='If activated plots of example a values distribution '
                                                                 'will be saved')
    args = parser.parse_args()

    sys.exit(main(args) or 0)
