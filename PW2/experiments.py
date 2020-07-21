import numpy as np
from plot_trajectory_2d import plot_levels, plot_trajectory
from oracles import QuadraticOracle
from oracles import create_log_reg_oracle, hess_vec_finite_diff
from optimization import lbfgs, hessian_free_newton, conjugate_gradients, gradient_descent
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


def setup_logging():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s %(message)s', level=logging.INFO, datefmt='%I:%M:%S')
    logging.getLogger().setLevel(logging.INFO)


def experiment_1():
    np.random.seed(31415)
    kappas = np.linspace(1., 30000., 50)
    plt.figure()
    colors = ['red', 'green', 'blue', 'orange']
    labels = ['n = 2', 'n = 10', 'n = 100', 'n = 1000']

    fig, ax = plt.subplots()

    repeat_times = 100
    runs = pd.DataFrame()
    for i, n in enumerate([2, 10, 100, 1000]):
        for j in tqdm(range(repeat_times)):
            iter_num_arr = []
            for kappa in kappas:
                D = np.random.uniform(1, kappa, n)
                D[0] = 1
                D[-1] = kappa
                A = diags(D)
                b = np.random.uniform(-1, 1, (n,))
                x_0 = np.zeros((n,))
                x_opt, message, history = conjugate_gradients(lambda x: A @ x, b, x_0, trace=True, max_iter=10**6)
                iter_num = len(history['time'])
                iter_num_arr.append(iter_num)

            runs['run_%s' % j] = iter_num_arr

        means = runs.mean(axis=1)
        stds = runs.std(axis=1)

        ax.plot(kappas, means, color=colors[i], label=labels[i], alpha=1.0)
        ax.fill_between(kappas, means - stds, means + stds, color=colors[i], alpha=0.3)
        ax.plot(kappas, means - stds, color=colors[i], alpha=0.7, linewidth=0.1)
        ax.plot(kappas, means + stds, color=colors[i], alpha=0.7, linewidth=0.1)

    ax.legend()
    ax.grid()
    ax.set_xlabel('Число обуcловленности')
    ax.set_ylabel('Число итераций до сходимости')
    os.makedirs("report/pics/1", exist_ok=True)
    fig.savefig("report/pics/1/CG_iterations_vs_condition_number.pdf", bbox_inches='tight')


def experiment_2():
    np.random.seed(31415)

    A, b = load_svmlight_file('data/gisette_scale')
    m = b.size
    oracle = create_log_reg_oracle(A, b, 1.0 / m, "optimized")
    x_0 = np.zeros((A.shape[1],))

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    ax1.set_yscale('log')
    ax1.set_xlabel('Номер итерации')
    ax1.set_ylabel('Относительный квадрат нормы градиента')
    ax1.grid()

    ax2.set_yscale('log')
    ax2.set_xlabel('Время от начала эксперимента')
    ax2.set_ylabel('Относительный квадрат нормы градиента')
    ax2.grid()

    for memory_size in tqdm([0, 1, 5, 10, 50, 100]):
        x_opt, message, history = lbfgs(oracle, x_0, trace=True, memory_size=memory_size)
        relative_grad_norms = (history['grad_norm'] / history['grad_norm'][0]) ** 2
        iter_times = history['time']
        iters = range(len(history['time']))
        if len(relative_grad_norms) > 400:
            relative_grad_norms = [elem for (i, elem) in enumerate(relative_grad_norms) if i % 2 == 0]
            iter_times = [elem for (i, elem) in enumerate(iter_times) if i % 2 == 0]
            iters = range(0, len(history['time']), 2)
        ax1.plot(iters, relative_grad_norms, label=f"l={memory_size}")
        ax2.plot(iter_times, relative_grad_norms, label=f"l={memory_size}")
        print(f"При l={memory_size} до сходимости потребовалось "
              f"{len(history['time'])} итераций и {history['time'][-1]} секунд")
    ax1.legend()
    ax2.legend()

    os.makedirs("report/pics/2", exist_ok=True)
    fig1.savefig("report/pics/2/lbfgs_grad_norm_vs_iter.pdf", bbox_inches='tight')
    fig2.savefig("report/pics/2/lbfgs_grad_norm_vs_time.pdf", bbox_inches='tight')


def experiment_3():
    np.random.seed(31415)
    data_path = "data"
    datasets = ["w8a", "gisette_scale", "real-sim", "news20", "rcv1_train"]
    for dataset in datasets:

        print("___________________________")
        logging.info(f"{dataset} is in process...")

        A, b = load_svmlight_file(os.path.join(data_path, dataset))
        print(A.shape, 1 - A.size / (A.shape[0] * A.shape[1]))

        m = b.size
        oracle = create_log_reg_oracle(A, b, 1.0 / m, "optimized")
        x_0 = np.zeros((A.shape[1],))
        x_opt1, message, history1 = gradient_descent(oracle, x_0, trace=True)
        logging.info("GD ended")
        x_opt2, message, history2 = hessian_free_newton(oracle, x_0, trace=True)
        logging.info("HFN ended")
        x_opt3, message, history3 = lbfgs(oracle, x_0, trace=True)
        logging.info("L-BFGS ended")

        os.makedirs("report/pics/3", exist_ok=True)

        plt.figure()
        plt.plot(history1['func'], label='GD')
        plt.plot(history2['func'], label='HFN')
        plt.plot(history3['func'], label='L-BFGS')

        print(f"GD iterations={len(history1['func'])}, time={history1['time'][-1]}")
        print(f"HFN iterations={len(history2['func'])}, time={history2['time'][-1]}")
        print(f"L-BFGS iterations={len(history3['func'])}, time={history3['time'][-1]}")

        plt.xlabel('Номер итерации')
        plt.ylabel('Значение функции потерь')
        plt.legend()
        plt.grid()
        plt.savefig(f"report/pics/3/logreg_loss_value_vs_iter_{dataset}.pdf", bbox_inches='tight')

        plt.figure()
        plt.plot(history1['time'], history1['func'], label='GD')
        plt.plot(history2['time'], history2['func'], label='HFN')
        plt.plot(history3['time'], history3['func'], label='L-BFGS')
        plt.xlabel('Время от начала эксперимента в секундах')
        plt.ylabel('Значение функции потерь')
        plt.legend()
        plt.grid()
        plt.savefig(f"report/pics/3/logreg_loss_value_vs_time_{dataset}.pdf", bbox_inches='tight')

        plt.figure()
        plt.plot(history1['time'], (history1['grad_norm'] / history1['grad_norm'][0]) ** 2, label='GD')
        plt.plot(history2['time'], (history2['grad_norm'] / history2['grad_norm'][0]) ** 2, label='HFN')
        plt.plot(history3['time'], (history3['grad_norm'] / history3['grad_norm'][0]) ** 2, label='L-BFGS')
        plt.yscale('log')
        plt.xlabel('Время от начала эксперимента в секундах')
        plt.ylabel('Относительный квадрат нормы градиента')
        plt.legend()
        plt.grid()
        plt.savefig(f"report/pics/3/logreg_grad_norm_vs_time_{dataset}.pdf", bbox_inches='tight')


def check_hess_vec():
    m, n = 1000, 500
    A = np.random.randn(m, n)
    b = np.sign(np.random.randn(m))
    regcoef = 1 / m

    x = np.random.randn(n)
    v = np.random.randn(n)

    logreg_oracle = create_log_reg_oracle(A, b, regcoef, oracle_type='optimized')

    v1 = logreg_oracle.hess_vec(x, v)
    v2 = hess_vec_finite_diff(logreg_oracle.func, x, v, eps=1e-6)
    res = np.allclose(v1, v2, atol=1e-2, rtol=1e-1)
    print(v1[:10])
    print(v2[:10])
    if res:
        print("Logreg hess_vec is OK!")
    else:
        print("Something wrong.")
    return res


def check_lyoha():
    B = np.array(range(1, 10)).reshape(3, 3)
    #         A = B@B.T
    A = B.dot(B.T)
    b = np.array(range(1, 4))
    oracle = create_log_reg_oracle(A, b, 1.0, oracle_type='optimized')

    x0 = np.zeros(3)
    x_expected = np.array([0.01081755, 0.02428744, 0.03775733])
    hist_expected = {'func': [0.69314718055994518, 0.060072133470449901, 0.020431219493905504],
                     'time': [0.0] * 3,
                     'grad_norm': [176.70314088889307, 3.295883082719103, 1.101366262174557]}

    x_star, msg, history = hessian_free_newton(oracle, x0, trace=True,
                                               line_search_options={'method': 'Wolfe'}, tolerance=1e-4)

    oracle.hess(np.zeros(3))
    oracle.hess(np.array([0.01952667, 0.04648169, 0.07343672]))
    pass


if __name__ == "__main__":

    check_lyoha()

    parser = ArgumentParser()
    parser.add_argument("--number", type=int, default=None)
    parser.add_argument("--check_hess_vec", type=bool, default=False)
    args = parser.parse_args()
    setup_logging()

    if args.number == 1:
        experiment_1()

    if args.number == 2:
        experiment_2()

    if args.number == 3:
        experiment_3()

    if args.check_hess_vec:
        print(check_hess_vec())

    if args.number is None:
        experiment_1()
        experiment_2()
        experiment_3()
