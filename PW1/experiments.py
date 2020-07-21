import numpy as np
from plot_trajectory_2d import plot_levels, plot_trajectory
from oracles import QuadraticOracle
from oracles import create_log_reg_oracle
from optimization import gradient_descent, newton
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import diags
from sklearn.datasets import load_svmlight_file
import pandas as pd
from scipy.stats import ortho_group
import matplotlib as mpl
from tqdm import tqdm
from argparse import ArgumentParser
import logging

def setup_logging():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s %(message)s', level=logging.INFO, datefmt='%I:%M:%S')
    logging.getLogger().setLevel(logging.INFO)


def run_trajectory(x_0, A, b=np.zeros((2,)), tag=""):
    oracle = QuadraticOracle(A, b)
    func = oracle.func
    line_search_options = [{'method': 'Constant', 'c': 0.1},
                           {'method': 'Constant', 'c': 1.0},
                           {'method': 'Armijo'},
                           {'method': 'Wolfe'}]

    Path("pics/" + tag).mkdir(parents=True, exist_ok=True)
    for options in line_search_options:
        x_opt, message, history = gradient_descent(oracle, x_0,
                                                   trace=True,
                                                   line_search_options=options)
        plt.figure()
        plot_levels(func)
        plot_trajectory(func, history['x'])

        filename = "pics/{2}/{0}{1}.png".format(options['method'],
                                             "_" + str(options['c']) if 'c' in options else "",
                                             tag)
        print("{}: {}".format(options['method'], len(history['x'])))
        plt.savefig(filename)


def experiment_1():
    A = np.diag([1., 1.2])
    x_0 = np.array([4., 1.])
    run_trajectory(x_0, A, tag="traj_low_cond_(x_0=(4,1))")

    A = np.diag([1., 10.])
    x_0 = np.array([4., 1])
    run_trajectory(x_0, A, tag="traj_high_cond_(x_0=(4,1))")

    A = np.diag([1., 1.2])
    x_0 = np.array([1., 4.])
    run_trajectory(x_0, A, tag="traj_low_cond_(x_0=(1,4))")

    A = np.diag([1., 10.])
    x_0 = np.array([1., 4.])
    run_trajectory(x_0, A, tag="traj_high_cond_(x_0=(1,4))")


def experiment_2():
    np.random.seed(31415)
    kappas = np.linspace(1., 1000., 50)
    plt.figure()
    colors = ['red', 'green', 'blue', 'orange']
    labels = ['n = 2', 'n = 10', 'n = 100', 'n = 1000']

    repeat_times = 100
    runs = pd.DataFrame()
    for i, n in enumerate([2, 10, 100, 1000]):
        for j in range(repeat_times):
            iter_num_arr = []
            for kappa in kappas:
                D = np.random.uniform(1, kappa, n)
                D[0] = 1
                D[-1] = kappa
                A = diags(D)
                b = np.random.uniform(-1, 1, (n,))
                x_0 = np.zeros((n,))
                oracle = QuadraticOracle(A, b)
                x_opt, message, history = gradient_descent(oracle, x_0, trace=True, max_iter=10**6)
                iter_num = len(history['func'])
                iter_num_arr.append(iter_num)

            runs['run_%s' % j] = iter_num_arr

        means = runs.mean(axis=1)
        stds = runs.std(axis=1)

        plt.plot(kappas, means, color=colors[i], label=labels[i], alpha=1.0)
        plt.fill_between(kappas, means - stds, means + stds, color=colors[i], alpha=0.3)
        plt.plot(kappas, means - stds, color=colors[i], alpha=0.7, linewidth=0.1)
        plt.plot(kappas, means + stds, color=colors[i], alpha=0.7, linewidth=0.1)

    plt.legend()
    plt.grid()
    plt.xlabel('Число обуcловленности')
    plt.ylabel('Число итераций до сходимости')
    plt.savefig("pics/iterations_vs_condition_number.pdf")


def experiment_3():
    np.random.seed(31415)
    m, n = 10000, 8000
    A = np.random.randn(m, n)
    b = np.sign(np.random.randn(m))
    regcoef = 1 / m

    oracle1 = create_log_reg_oracle(A, b, regcoef, oracle_type='usual')
    oracle2 = create_log_reg_oracle(A, b, regcoef, oracle_type='optimized')

    x_0 = np.zeros((n,))
    x_opt1, message, history1 = gradient_descent(oracle1, x_0, trace=True)
    x_opt2, message, history2 = gradient_descent(oracle2, x_0, trace=True)
    print(x_opt1, x_opt2)

    plt.figure()
    plt.plot(history1['func'], label='Usual')
    plt.plot(history2['func'], label='Optimized')
    plt.xlabel('Номер итерации')
    plt.ylabel('Значение функции потерь')
    plt.legend()
    plt.grid()
    plt.savefig("pics/logreg_values")

    plt.figure()
    plt.plot(history1['time'], history1['func'], label='Usual')
    plt.plot(history2['time'], history2['func'], label='Optimized')
    plt.xlabel('Время от начала эксперимента в секундах')
    plt.ylabel('Значение функции потерь')
    plt.legend()
    plt.grid()
    plt.savefig("pics/logreg_loss_value_vs_time")

    plt.figure()
    plt.plot(history1['time'], 2 * np.log((history1['grad_norm'] / history1['grad_norm'][0])), label='Usual')
    plt.plot(history2['time'], 2 * np.log((history2['grad_norm'] / history2['grad_norm'][0])), label='Optimized')
    plt.xlabel('Время от начала эксперимента в секундах')
    plt.ylabel('Логарифм относительного квадрата нормы градиента')
    plt.legend()
    plt.grid()
    plt.savefig("pics/logreg_grad_norm_vs_time")


def experiment_4():
    path = 'data'
    np.random.seed(31415)
    datasets = ["w8a", "gisette_scale", "real-sim"]
    for dataset in datasets:
        A, b = load_svmlight_file(path + '/' + dataset)
        m = b.size
        oracle = create_log_reg_oracle(A, b, 1.0 / m, "optimized")
        x_0 = np.zeros((A.shape[1],))
#        x_opt1, message, history1 = gradient_descent(oracle, x_0, trace=True)
        if dataset != 'real-sim':
            x_opt2, message, history2 = newton(oracle, x_0, trace=True)
        print(len(history2['time']), history2['time'][-1])
        continue
        
        plt.figure()
        plt.plot(history1['time'], history1['func'], label='GD')
        if dataset != 'real-sim':
            plt.plot(history2['time'], history2['func'], label='Newton')
        plt.xlabel('Время от начала эксперимента в секундах')
        plt.ylabel('Значение функции потерь')
        plt.legend()
        plt.grid()
        plt.savefig("pics/logreg_loss_value_vs_time_" + dataset)

        plt.figure()
        plt.plot(history1['time'], 2 * np.log((history1['grad_norm'] / history1['grad_norm'][0])), label='GD')
        if dataset != 'real-sim':
            plt.plot(history2['time'], 2 * np.log((history2['grad_norm'] / history2['grad_norm'][0])), label='Newton')
        plt.xlabel('Время от начала эксперимента в секундах')
        plt.ylabel('Логарифм относительного квадрата нормы градиента')
        plt.legend()
        plt.grid()
        plt.savefig("pics/logreg_grad_norm_vs_time_" + dataset)


def experiment_5_and_6(algo='gd'):
    np.random.seed(31415)
    m, n = 2000, 1000
    A = np.random.randn(m, n)
    b = np.sign(np.random.randn(m))
    regcoef = 1 / m


    logreg_oracle = create_log_reg_oracle(A, b, regcoef, oracle_type='optimized')

    line_search_options = [
        {'method': 'Constant', 'c': 1.0},
        {'method': 'Constant', 'c': 0.95},
        {'method': 'Constant', 'c': 0.9},
        {'method': 'Constant', 'c': 0.85},
        {'method': 'Armijo', 'c1': 1e-8},
        {'method': 'Armijo', 'c1': 1e-6},
        {'method': 'Armijo', 'c1': 1e-4},
        {'method': 'Armijo', 'c1': 1e-1},
        {'method': 'Wolfe', 'c2': 1.5},
        {'method': 'Wolfe', 'c2': 0.9},
        {'method': 'Wolfe', 'c2': 0.1},
        {'method': 'Wolfe', 'c2': 0.01},
    ]

    colors = ['#e66101', '#fdb863', '#b2abd2', '#5e3c99']
    styles = {'Constant': {'linestyle': '--',
                         'dashes': (2, 5),
                         'linewidth': 2},
              'Armijo': {'linestyle': '--',
                         'dashes': (5, 2)},
              'Wolfe': {'linestyle': 'solid'},
              }

    x_0_list = [None] * 3
    x_0_list[0] = np.zeros((n,))
    x_0_list[1] = np.random.uniform(-1, 1, (n,))
    x_0_list[2] = np.ones((n,))

    for k, x_0 in enumerate(x_0_list):
        plt.figure(figsize=(12, 9))
        for i, options in tqdm(enumerate(line_search_options)):
            if algo == 'GD':
                x_opt, message, history = gradient_descent(logreg_oracle, x_0,
                                                           trace=True,
                                                           line_search_options=options)
            else:
                x_opt, message, history = newton(logreg_oracle, x_0,
                                                 trace=True,
                                                 line_search_options=options)
            args = list(options.keys()) + list(options.values())
            label = "{} ({}={}, {}={})".format(algo, args[0], args[2], args[1], args[3])
            values = 2 * np.log((history['grad_norm'] / history['grad_norm'][0]))
            method = args[2]
            plt.plot(values + np.random.randn(values.size) * 0.05, color=colors[i % len(colors)],
                     label=label, alpha=0.7, **styles[method])
        plt.xlabel('Номер итерации')
        plt.ylabel('Логарифм относительного квадрата нормы градиента')
        plt.legend(loc='upper right')
        plt.grid()

        Path("pics/logreg_{}_linear_search_strategies".format(algo)).mkdir(parents=True, exist_ok=True)
        plt.savefig("pics/logreg_{}_linear_search_strategies/x_0_{}.png".format(algo, k))

    np.random.seed(31415)
    n = 2000
    C = ortho_group.rvs(n)
    A = C.T @ np.diag(np.random.uniform(1, 20, (n,))) @ C
    b = np.random.randn(n)
    x_0 = np.zeros((n,))

    quadratic_oracle = QuadraticOracle(A, b)
    x_opt = np.linalg.solve(A, b)
    f_opt = quadratic_oracle.func(x_opt)

    line_search_options = [
                           {'method': 'Constant', 'c': 0.09},
                           {'method': 'Constant', 'c': 0.085},
                           {'method': 'Constant', 'c': 0.08},
                           {'method': 'Constant', 'c': 0.075},
                           {'method': 'Armijo', 'c1': 1e-10},
                           {'method': 'Armijo', 'c1': 1e-7},
                           {'method': 'Armijo', 'c1': 1e-4},
                           {'method': 'Armijo', 'c1': 1e-1},
                           {'method': 'Wolfe', 'c2': 1.5},
                           {'method': 'Wolfe', 'c2': 0.9},
                           {'method': 'Wolfe', 'c2': 0.1},
                           {'method': 'Wolfe', 'c2': 0.01},
    ]

    x_0_list = [None] * 3
    x_0_list[0] = np.zeros((n,))
    x_0_list[1] = np.random.uniform(-1, 1, (n,))
    x_0_list[2] = x_opt + np.random.randn(n,) * 0.2

    for k, x_0 in enumerate(x_0_list):
        plt.figure(figsize=(12, 9))
        for i, options in tqdm(enumerate(line_search_options)):
            if algo == 'GD':
                x_opt, message, history = gradient_descent(quadratic_oracle, x_0,
                                                           trace=True,
                                                           line_search_options=options)
            else:
                x_opt, message, history = newton(quadratic_oracle, x_0,
                                                 trace=True,
                                                 line_search_options=options)
            args = list(options.keys()) + list(options.values())
            label = "{} ({}={}, {}={})".format(algo, args[0], args[2], args[1], args[3])
            values = np.log(np.abs((history['func'] - f_opt) / f_opt) + 1e-16)
            method = args[2]
            plt.plot(values + np.random.randn(values.size) * 0.05, color=colors[i % len(colors)],
                     label=label, alpha=0.7, **styles[method])
        plt.xlabel('Номер итерации')
        plt.ylabel('Логарифм относительной невязки')
        plt.legend(loc='upper right')
        plt.grid()
        Path("pics/quadratic_{}_linear_search_strategies".format(algo)).mkdir(parents=True, exist_ok=True)
        plt.savefig("pics/quadratic_{}_linear_search_strategies/x_0_{}.png".format(algo, k))

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
        experiment_3()

    if args.number == 3:
        experiment_4()

    if args.number == 5:
        experiment_5_and_6(algo='GD')

    if args.number == 6:
        experiment_5_and_6(algo='Newton')
