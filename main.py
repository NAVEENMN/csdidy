#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
import os
import glob
import time
import json
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI
import seaborn as sns
from sklearn import metrics
from multiprocessing import Pool
from experiment import Experiment

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description='Control variables for PC Algorithm.')
parser.add_argument('--path', default='/Users/naveenmysore/Documents/data/csdi_data')
parser.add_argument('--pv', default=0.02, help='Threshold p value', type=float)
parser.add_argument('--tau_min', default=1, help='Tau min', type=int)
parser.add_argument('--tau_max', default=5, help='Tau max', type=int)
args = parser.parse_args()


path = os.path.join(os.getcwd(), 'meta.json')
meta_data = dict()
if os.path.exists(path):
    with open(path) as json_file:
        meta_data = json.load(json_file)
exp_id = meta_data['recent']

experiment = Experiment()
experiment.set_id(id=exp_id)
experiment.load_settings()


# Variables of interest
particle_vars, spring_vars = experiment.get_vars()
variables_dim_1 = [var for var in particle_vars if 'x_position' in var]
variables_dim_2 = [var for var in particle_vars if 'y_position' in var]

data_observations_path = os.path.join(args.path, f'observations_{experiment.get_id()}.csv')
springs_observations_path = os.path.join(args.path, f'springs_{experiment.get_id()}.csv')


def load_observations(path, _variables):
    data = pd.read_csv(path)
    # reversing the data so that the least valued
    # 0 index represents most reset observation
    # collect variables of interest
    data = data[::-1]
    if _variables:
        data = data[_variables]
    print(data.head())
    var_names = data.columns.values
    dataframe = pp.DataFrame(data.values,
                             datatime=np.arange(len(data)),
                             var_names=var_names)
    return dataframe


# Positions of particles
observations_dim_1 = load_observations(data_observations_path, variables_dim_1)
observations_dim_2 = load_observations(data_observations_path, variables_dim_2)

# Spring constants between particles
springs = load_observations(springs_observations_path, [])


def setup_pcmci(data_frame):
    pcmci = PCMCI(dataframe=data_frame,
                  cond_ind_test=ParCorr(),
                  verbosity=0)
    # Auto correlations are helpful to determine tau
    # correlations = pcmci.get_lagged_dependencies(tau_max=100, val_only=True)['val_matrix']
    return pcmci


def get_auroc(time_step, p_values_dim_1, p_values_dim_2, _springs):
    _vars = [f'particle_{i}' for i in range(len(variables_dim_1))]
    true_labels = []
    predictions = []
    for p_a in range(len(_vars)):
        for p_b in range(len(_vars)):
            if p_a != p_b:
                if _springs.iloc[time_step][f's_{p_a}_{p_b}'] != 0.0:
                    # has a springs
                    true_labels.append(1)
                else:
                    true_labels.append(0)
                avg_p_val = 1 - ((p_values_dim_1[p_a][p_b][time_step] + p_values_dim_2[p_a][p_b][time_step]) / 2.0)
                predictions.append(avg_p_val)

    #conf_matrix = metrics.confusion_matrix(true_labels, predictions)
    #print("** CONF Matrix **")
    #print(conf_matrix)
    #print(true_labels)
    #print(predictions)

    fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions)
    auroc = metrics.auc(fpr, tpr)
    args = parser.parse_args()

    details = f'p_threshold: {args.pv}, taumin: {args.tau_min},, taumax: {args.tau_max}, fpr: {round(np.mean(fpr), 2)}, tpr: {round(np.mean(tpr), 2)}, auroc:{round(auroc, 2)}'
    return tpr, fpr, auroc, details


def construct_causal_graph(time_step, p_values_dim_1, p_values_dim_2, p_threshold, _springs):
    _vars = [f'particle_{i}' for i in range(len(variables_dim_1))]
    graph = nx.complete_graph(_vars)
    for p_a in range(len(_vars)):
        for p_b in range(len(_vars)):
            if p_a != p_b:
                avg_p_val = (p_values_dim_1[p_a][p_b][time_step] + p_values_dim_2[p_a][p_b][time_step])/2.0
                # closer to 0 -> add link
                # closer to 1 -> remove link
                if graph.has_edge(f'particle_{p_a}', f'particle_{p_b}') and (np.abs(avg_p_val) > p_threshold):
                    graph.remove_edge(f'particle_{p_a}', f'particle_{p_b}')

    # variables_dim_1 is ok
    tpr, fpr, auroc, details = get_auroc(time_step, p_values_dim_1, p_values_dim_2, _springs)
    save_graph(time_step, graph, variables_dim_1, details)
    return auroc, graph


def save_graph(time_step, causal_graph, _variables, details):
    # observations -> positions
    # springs -> spring constants
    # causal graph from predictions
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))

    # ----- Plotting Particle positions
    axes[0][0].set_title('Particle position')
    entries = []
    _observations = pd.read_csv(data_observations_path)
    for particle_id in range(0, experiment.get_numb_of_particles()):
        data = {'particle': particle_id,
                'x_cordinate': _observations.iloc[time_step][f'p_{particle_id}_x_position'],
                'y_cordinate': _observations.iloc[time_step][f'p_{particle_id}_y_position']}
        entries.append(data)
    pdframe = pd.DataFrame(entries)
    pl = sns.scatterplot(data=pdframe,
                         x='x_cordinate',
                         y='y_cordinate',
                         hue='particle',
                         ax=axes[0][0])
    pl.set_ylim(-5.0, 5.0)
    pl.set_xlim(-5.0, 5.0)

    _springs = pd.read_csv(springs_observations_path)
    # ----- Plotting spring constants

    _springs = pd.read_csv(springs_observations_path)
    axes[0][1].set_title(f'Spring connections')
    columns = [f'particle_{i}' for i in range(len(_variables))]
    s_mat = np.zeros(shape=(len(_variables), len(_variables)))
    npr = experiment.get_numb_of_particles()
    for p_a in range(npr):
        for p_b in range(npr):
            if p_a != p_b:
                s_mat[p_a][p_b] = _springs.iloc[time_step][f's_{p_a}_{p_b}']
    s_mat = np.reshape(s_mat, (npr, npr))
    sns.heatmap(pd.DataFrame(s_mat, columns=columns, index=columns),
                vmin=0.0, vmax=2.0, ax=axes[0][1])


    # ----- Plotting Ground Truth Causal graph
    axes[1][0].set_title(f'Ground truth causal graph (Springs)')
    _vars = [f'particle_{i}' for i in range(len(_variables))]
    graph = nx.complete_graph(_vars)
    for p_a in range(len(_vars)):
        for p_b in range(len(_vars)):
            if p_a != p_b and np.abs(_springs.iloc[time_step][f's_{p_a}_{p_b}']) == 0.0 and graph.has_edge(f'particle_{p_a}', f'particle_{p_b}'):
                graph.remove_edge(f'particle_{p_a}', f'particle_{p_b}')
    nx.draw(graph,
            pos=nx.circular_layout(graph),
            with_labels=True,
            ax=axes[1][0],
            node_size=500)

    # ----- Plotting Predicted Causal graph
    axes[1][1].set_title(f'Predicted causal graph (Springs)')
    nx.draw(causal_graph,
            pos=nx.circular_layout(causal_graph),
            with_labels=True,
            ax=axes[1][1],
            node_size=500)

    fig.suptitle(f'Time step {time_step} - {details}')

    #plt.show()
    fig.savefig(os.path.join(os.getcwd(), 'tmp', f'graph_{time_step}.png'))
    plt.clf()
    plt.close(fig)


def get_parents(tau_max, tau_min):
    _vars = list(range(len(variables_dim_1)))
    _lags = list(range(-(tau_max), -tau_min + 1, 1))
    # Set the default as all combinations of the selected variables
    _int_sel_links = {}
    for j in _vars:
        _int_sel_links[j] = [(var, -lag) for var in _vars
                             for lag in range(tau_min, tau_max + 1)
                             if not (var == j and lag == 0)]
    # Remove contemporary links
    for j in _int_sel_links.keys():
        _int_sel_links[j] = [link for link in _int_sel_links[j]
                             if link[1] != 0]
    # Remove self links
    for j in _int_sel_links.keys():
        _int_sel_links[j] = [link for link in _int_sel_links[j]
                             if link[0] != j]

    return _int_sel_links


def run_pc(dim):
    parents = get_parents(tau_min=args.tau_min, tau_max=args.tau_max)
    if dim == 1:
        logging.info(f'Running PCMCI on dim {dim}')
        pcmci = setup_pcmci(observations_dim_1)
    else:
        logging.info(f'Running PCMCI on dim {dim}')
        pcmci = setup_pcmci(observations_dim_2)

    pcmci.verbosity = 0
    results = pcmci.run_pcmci(tau_max=args.tau_max,
                              tau_min=args.tau_min,
                              selected_links=parents)
    p_values = results['p_matrix'].round(3)
    logging.info(f'Saving pcmci {dim}')
    np.save(os.path.join(os.getcwd(), 'data', f'p_values_dim_{dim}'), p_values)
    logging.info(f'Saved pcmci {dim}')


def plot_link_distribution(links_distribution):
    fp_in = f"{os.getcwd()}/result/*.png"
    for f in glob.glob(fp_in):
        os.remove(f)
    fp_in = f"{os.getcwd()}/result/*.json"
    for f in glob.glob(fp_in):
        os.remove(f)
    df = pd.DataFrame(links_distribution)
    experiment.load_results()
    _title = f'LinkDistribution - {experiment.get_id()} - tau:{args.tau_max} - p_threshold: {experiment.get_p_threshold()}'
    dist_plt = sns.histplot(df, x='links').set_title(_title)
    dist_plt = dist_plt.get_figure()
    dist_plt.savefig(f'result/links_dist_{experiment.get_id()}.png')


def main():
    # First they estimate all parents for last layer.
    # Using the same kin relationship as parent sets
    # The same set of parents are used for momemtary ci test backwards in time.
    # Running pcmci on dim 1

    exp_id = experiment.get_id()

    # *** Control Variables ***
    tau_max = args.tau_max
    p_threshold = args.pv

    _springs = pd.read_csv(springs_observations_path)

    # Clean before running
    file_list = glob.glob(os.path.join('data', "*"))
    for fl in file_list:
        os.remove(fl)
    file_list = glob.glob(os.path.join('media', "*"))
    for fl in file_list:
        os.remove(fl)
    file_list = glob.glob(os.path.join('result', "*"))
    for fl in file_list:
        os.remove(fl)

    # Results are published on independent threads.
    start_time = time.time()
    #dims = [1, 2]
    #with Pool(4) as p:
    #    p.map(run_pc, dims)
    run_pc(dim=1)
    run_pc(dim=2)

    with open('data/p_values_dim_1.npy', 'rb') as f1:
        p_values_dim_1 = np.load(f1)
    with open('data/p_values_dim_2.npy', 'rb') as f2:
        p_values_dim_2 = np.load(f2)

    logging.info('Constructing causal graph')
    if not os.path.exists(f"{os.getcwd()}/tmp"):
        os.mkdir(f"{os.getcwd()}/tmp")
    links_distribution = dict()
    links_distribution['links'] = []
    _vars = [f'particle_{i}' for i in range(len(variables_dim_1))]
    time_step = tau_max-1
    aurocs = []
    while time_step != 0:
        auroc, graph = construct_causal_graph(time_step, p_values_dim_1, p_values_dim_2, p_threshold, _springs)
        aurocs.append(auroc)
        for p_a in range(len(_vars)):
            for p_b in range(len(_vars)):
                if (p_a != p_b) and (graph.has_edge(f'particle_{p_a}', f'particle_{p_b}')):
                    links_distribution['links'].append(f'{p_a}-{p_b}')
        time_step -= 1
    end_time = time.time()

    print('Done.')
    print(f'Average AUROC {np.mean(aurocs)}')

    # Publish results
    # Read simulation settings
    with open('experiment.json') as json_file:
        exp = json.load(json_file)
    exp[exp_id]['results']['conducted'] = True
    exp[exp_id]['results']['tau'] = tau_max
    exp[exp_id]['results']['p_threshold'] = p_threshold
    exp[exp_id]['results']['auroc']['mean'] = np.mean(aurocs)
    exp[exp_id]['results']['auroc']['max'] = np.max(aurocs)
    exp[exp_id]['results']['auroc']['min'] = np.min(aurocs)
    exp[exp_id]['results']['auroc']['std'] = np.std(aurocs)

    # update settings used
    exp[exp_id]['settings']['tau_min'] = args.tau_min
    exp[exp_id]['settings']['tau_max'] = args.tau_max

    with open(f'experiment.json', 'w') as f:
        json.dump(exp, f, indent=4)

    # plot_link_distribution(links_distribution)

    print(f'Total time taken {end_time - start_time}')

# delete all tmp files.
fp_in = f"{os.getcwd()}/tmp/timestep_*.png"
for f in glob.glob(fp_in):
    os.remove(f)

logging.info('trajectory gif stores in media')


if __name__ == "__main__":
    main()