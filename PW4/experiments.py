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
from optimization import subgradient_method, proximal_gradient_method, proximal_fast_gradient_method
from matplotlib import rc
from oracles import lasso_duality_gap, create_lasso_prox_oracle, create_lasso_nonsmooth_oracle
from matplotlib import rcParams
from itertools import accumulate

rcParams["legend.loc"] = 'upper right'

def setup_logging():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s %(message)s', level=logging.INFO, datefmt='%I:%M:%S')
    logging.getLogger().setLevel(logging.INFO)


def experiment_1():

    np.random.seed(31415)

    data_path = "data"
    datasets = ["bodyfat", "housing"]

    alpha_grid = [1e-3, 1e-2, 1e-1, 1.0, 5.0]

    for dataset in datasets:
        print("___________________________")
        logging.info(f"{dataset} is in process...")

        A, b = load_svmlight_file(os.path.join(data_path, dataset))
        n = A.shape[1]

        x_grid = list()
        x_grid.append(np.zeros(A.shape[1]))
        x_grid.append(np.ones(A.shape[1]))
        x_grid.append(np.random.randn(n))

        for i, x0 in enumerate(x_grid):
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            fig2, ax2 = plt.subplots(figsize=(12, 8))

            ax1.set_xlabel('Номер итерации')
            ax1.set_ylabel('Гарантированная точность по зазору двойственности')
            ax1.grid()
            ax1.set_yscale('log')

            ax2.set_xlabel('Время от начала эксперимента')
            ax2.set_ylabel('Гарантированная точность по зазору двойственности')
            ax2.grid()
            ax2.set_yscale('log')

            oracle = create_lasso_nonsmooth_oracle(A, b, 1.0)
            # oracle = create_lasso_prox_oracle(A, b, 1.0)
            print(A.shape, 1 - A.size / (A.shape[0] * A.shape[1]), np.linalg.matrix_rank(A.toarray()))

            os.makedirs("report/pics/1", exist_ok=True)

            for alpha in alpha_grid:
                x_opt, message, history = subgradient_method(oracle, x0, alpha_0=alpha, trace=True, max_iter=1000000)
                ax1.plot(history['duality_gap'], label=f'alpha={alpha}')
                ax2.plot(history['time'], history['duality_gap'], label=f'alpha={alpha}')

            ax1.legend()
            ax2.legend()

            os.makedirs("report/pics/1", exist_ok=True)
            fig1.savefig(f"report/pics/1/subgradient_gap_vs_iter_alpha_x_{i}_{dataset}.pdf", bbox_inches='tight')
            fig2.savefig(f"report/pics/1/subgradient_gap_vs_time_alpha_x_{i}_{dataset}.pdf", bbox_inches='tight')


def experiment_2():

    np.random.seed(31415)

    data_path = "data"
    datasets = ["bodyfat", "housing"]

    for dataset in datasets:
        print("___________________________")
        logging.info(f"{dataset} is in process...")

        A, b = load_svmlight_file(os.path.join(data_path, dataset))

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.set_xlabel('Номер итерации')
        ax.set_ylabel('Суммарное число шагов подбора L')
        ax.grid()

        oracle = create_lasso_prox_oracle(A, b, 1.0)
        print(A.shape, 1 - A.size / (A.shape[0] * A.shape[1]), np.linalg.matrix_rank(A.toarray()))

        n = A.shape[1]
        x0 = np.random.randn(n)
        x_opt, message, history_usual = proximal_gradient_method(oracle, x0, trace=True, max_iter=10000)
        x_opt, message, history_fast = proximal_fast_gradient_method(oracle, x0, trace=True, max_iter=10000)

        sum_int_steps_usual = list(accumulate(history_usual['int_steps']))
        sum_int_steps_fast = list(accumulate(history_fast['int_steps']))

        ax.plot(sum_int_steps_usual, label="Usual proximal method")
        ax.plot(sum_int_steps_fast, label=f'Fast proximal method')

        ax.legend()

        os.makedirs("report/pics/2", exist_ok=True)
        fig.savefig(f"report/pics/2/prox_methods_steps_{dataset}.pdf", bbox_inches='tight')


def experiment_3(parameter='n', method='subgradient', seed=31415):

    # n, m and lambda
    n, m, reg_coef = 500, 500, 1.0

    np.random.seed(seed)
    grids = dict()
    grids['n'] = [10, 100, 1000]
    grids['m'] = [10, 100, 1000]
    grids['reg_coef'] = [0.01, 0.1, 1.0, 10.0]

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
    os.makedirs("report/pics/3", exist_ok=True)

    experiment_parameters = {'n': n, 'm': m, 'reg_coef': 1.0}

    for value in grids[parameter]:
        experiment_parameters[parameter] = value
        A = np.random.randn(experiment_parameters['m'], experiment_parameters['n'])
        b = np.random.randn(experiment_parameters['m'])
        x_0 = np.ones(experiment_parameters['n'])

        reg_coef = experiment_parameters['reg_coef']

        if method == 'subgradient':
            oracle = create_lasso_nonsmooth_oracle(A, b, reg_coef)
            x_opt, message, history = subgradient_method(oracle, x_0, trace=True, max_iter=10000)

        if method == 'proximal':
            oracle = create_lasso_prox_oracle(A, b, reg_coef)
            x_opt, message, history = proximal_gradient_method(oracle, x_0, trace=True, max_iter=10000)

        if method == 'proximal_fast':
            oracle = create_lasso_prox_oracle(A, b, reg_coef)
            x_opt, message, history = proximal_fast_gradient_method(oracle, x_0, trace=True, max_iter=10000)

        ax1.plot(history['duality_gap'], label=f'{parameter}={value}')
        ax2.plot(history['time'], history['duality_gap'], label=f'{parameter}={value}')

    ax1.legend()
    ax2.legend()

    os.makedirs(f"report/pics/3/{method}", exist_ok=True)
    fig1.savefig(f"report/pics/3/{method}/lasso_gap_vs_iter_{parameter}.pdf", bbox_inches='tight')
    fig2.savefig(f"report/pics/3/{method}/lasso_gap_vs_time_{parameter}.pdf", bbox_inches='tight')


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--number", type=int, default=None)
    args = parser.parse_args()
    setup_logging()

    if args.number == 1:
        experiment_1()

    if args.number == 2:
        experiment_2()

    if args.number == 3:
        for method in ['subgradient', 'proximal', 'proximal_fast']:
            experiment_3(parameter='n', method=method)
            experiment_3(parameter='m', method=method)
            experiment_3(parameter='reg_coef', method=method)

    if args.number is None:
        experiment_1()
        experiment_2()
        for method in ['subgradient', 'proximal', 'proximal_fast']:
            experiment_3(parameter='n', method=method)
            experiment_3(parameter='m', method=method)
            experiment_3(parameter='reg_coef', method=method)
