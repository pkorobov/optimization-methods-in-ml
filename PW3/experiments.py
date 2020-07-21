import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import diags
from sklearn.datasets import load_svmlight_file
import pandas as pd
from scipy.stats import ortho_group
import matplotlib as mpl
from tqdm import tqdm
from argparse import ArgumentParser
import os
import logging
from optimization import barrier_method_lasso
from matplotlib import rc
from oracles import lasso_duality_gap
from matplotlib import rcParams

rcParams["legend.loc"] = 'upper right'

def setup_logging():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s %(message)s', level=logging.INFO, datefmt='%I:%M:%S')
    logging.getLogger().setLevel(logging.INFO)


def experiment_1(parameter='gamma'):

    # gamma and eps_inner

    np.random.seed(31415)
    data_path = "data"
    datasets = ["abalone", "cpusmall", "housing", "triazines"]

    gamma_grid = [2, 10, 100, 1000]
    inner_eps_grid = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 5e-1]

    for dataset in datasets:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        ax1.set_xlabel('Номер итерации')
        ax1.set_ylabel('Гарантированная точность по зазору двойственности')
        ax1.grid()
        ax1.set_yscale('log')

        ax2.set_xlabel('Время от начала эксперимента')
        ax2.set_ylabel('Гарантированная точность по зазору двойственности')
        ax2.grid()
        ax2.set_yscale('log')

        print("___________________________")
        logging.info(f"{dataset} is in process...")

        A, b = load_svmlight_file(os.path.join(data_path, dataset))
        print(A.shape, 1 - A.size / (A.shape[0] * A.shape[1]), np.linalg.matrix_rank(A.toarray()))

        os.makedirs("report/pics/1", exist_ok=True)

        n = A.shape[1]
        x0 = np.zeros(n)
        u0 = 0.1 * np.ones(n)
        # x0 = np.random.randn(n)
        # u0 = np.abs(x0) * 2
        if parameter == 'gamma':
            for gamma in gamma_grid:
                (x_opt, u_opt), message, history = barrier_method_lasso(A, b, 1.0,
                                                                        x0, u0, gamma=gamma,
                                                                        lasso_duality_gap=lasso_duality_gap,
                                                                        trace=True)
                ax1.plot(history['duality_gap'], label=f'gamma={gamma}')
                ax2.plot(history['time'], history['duality_gap'], label=f'gamma={gamma}')
        else:
            for inner_eps in inner_eps_grid:
                (x_opt, u_opt), message, history = barrier_method_lasso(A, b, 1.0, np.zeros(n),
                                                                        np.ones(n), tolerance_inner=inner_eps,
                                                                        lasso_duality_gap=lasso_duality_gap,
                                                                        trace=True)
                ax1.plot(history['duality_gap'], label=f'eps_inner={inner_eps}')
                ax2.plot(history['time'], history['duality_gap'], label=f'eps_inner={inner_eps}')

        ax1.legend()
        ax2.legend()

        os.makedirs("report/pics/1", exist_ok=True)
        fig1.savefig(f"report/pics/1/barrier_lasso_gap_vs_iter_{parameter}_{dataset}.pdf", bbox_inches='tight')
        fig2.savefig(f"report/pics/1/barrier_lasso_gap_vs_time_{parameter}_{dataset}.pdf", bbox_inches='tight')


def experiment_2(parameter='n', seed=31415):

    # n, m and lambda
    n, m, reg_coef = 500, 500, 1.0

    np.random.seed(seed)
    grids = dict()
    grids['n'] = [10, 100, 500, 1000, 2000]
    grids['m'] = [10, 100, 500, 1000, 2000]
    grids['reg_coef'] = [0.01, 0.1, 1.0, 10.0, 100.0]

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    ax1.set_xlabel('Номер итерации')
    ax1.set_ylabel('Гарантированная точность по зазору двойственности')
    ax1.grid()
    ax1.set_yscale('log')

    ax2.set_xlabel('Время от начала эксперимента')
    ax2.set_ylabel('Гарантированная точность по зазору двойственности')
    ax2.grid()
    ax2.set_yscale('log')
    os.makedirs("report/pics/2", exist_ok=True)

    experiment_parameters = {'n': n, 'm': m, 'reg_coef': 1.0}

    for value in grids[parameter]:
        experiment_parameters[parameter] = value
        A = np.random.randn(experiment_parameters['m'], experiment_parameters['n'])
        b = np.random.randn(experiment_parameters['m'])
        # x_0 = np.zeros(experiment_parameters['n'])
        x_0 = 0.1 * np.ones(experiment_parameters['n'])
        u_0 = 2 * x_0

        reg_coef = experiment_parameters['reg_coef']
        (x_opt, u_opt), message, history = barrier_method_lasso(A, b, reg_coef, x_0=x_0, u_0=u_0,
                                                                lasso_duality_gap=lasso_duality_gap,
                                                                trace=True)
        ax1.plot(history['duality_gap'], label=f'{parameter}={value}')
        ax2.plot(history['time'], history['duality_gap'], label=f'{parameter}={value}')

    ax1.legend()
    ax2.legend()

    os.makedirs("report/pics/2", exist_ok=True)
    fig1.savefig(f"report/pics/2/barrier_lasso_gap_vs_iter_{parameter}.pdf", bbox_inches='tight')
    fig2.savefig(f"report/pics/2/barrier_lasso_gap_vs_time_{parameter}.pdf", bbox_inches='tight')


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--number", type=int, default=None)
    args = parser.parse_args()
    setup_logging()

    if args.number == 1:
        experiment_1(parameter='gamma')
        experiment_1(parameter='inner_eps')

    if args.number == 2:
        experiment_2(parameter='n')
        experiment_2(parameter='m')
        experiment_2(parameter='reg_coef')

    if args.number is None:
        experiment_1(parameter='gamma')
        experiment_1(parameter='inner_eps')
        experiment_2(parameter='n')
        experiment_2(parameter='m')
        experiment_2(parameter='reg_coef')
