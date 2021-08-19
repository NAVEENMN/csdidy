import os
import json
import seaborn as sns
import pandas as pd
import glob
import matplotlib.pyplot as plt
import argparse
from experiment import Experiment
from multiprocessing import Pool
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--op', type=str, default='trajectory',
                    help='What simulation to generate.')
parser.add_argument('--length', type=int, default=500,
                    help='Length of trajectory.')
parser.add_argument('--data_path', default='/Users/naveenmysore/Documents/data/csdi_data')
args = parser.parse_args()


def get_experiment_id():
    path = os.path.join(os.getcwd(), 'meta.json')
    meta_data = dict()
    if os.path.exists(path):
        with open(path) as json_file:
            meta_data = json.load(json_file)
    exp_id = meta_data['recent']
    return exp_id


def create_gif():
    import os
    import glob
    from PIL import Image

    loc = get_experiment_id()
    fcont = len(glob.glob(f"{os.getcwd()}/tmp/graph_*.png"))
    # ref: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f"{os.getcwd()}/tmp/graph_{i}.png") for i in range(1, fcont)]
    img.save(fp=f"{os.getcwd()}/media/{loc}.gif",
             format='GIF',
             append_images=imgs,
             save_all=True,
             duration=10,
             loop=0)

    # delete all png files.
    fp_in = f"{os.getcwd()}/tmp/graph_*.png"
    for f in glob.glob(fp_in):
        os.remove(f)


def generate_auroc_plots(cls):
    results_files = glob.glob('/Users/naveenmysore/Documents/plots/csdi/recent/*.json')
    auroc_results = dict()
    columns = ['trajectory_length', 'number_of_simulations', 'sample_frequency', 'num_of_particles', 'data_size',
                   'tau', 'p_threshold', 'auroc']
    auroc_results = {label: [] for label in columns}
    for json_file in results_files:
        with open(json_file) as jfile:
            jresults = json.load(jfile)
            for col_name in jresults.keys():
                auroc_results[col_name].append(jresults[col_name])
            print(jresults)
    df = pd.DataFrame(auroc_results)
    print(df.head())
    p = sns.lineplot(df.tau, df.auroc)
    fig = p.get_figure()
    fig.savefig('/Users/naveenmysore/Documents/tau_auroc.png')
    # plt.xscale('log')
    plt.show()

def plt_pair():
    exp_id = get_experiment_id()
    observations = pd.read_csv(f'/Users/naveenmysore/Documents/data/csdi_data/observations_{exp_id}.csv')
    print(observations.head())
    sns.pairplot(observations)
    plt.show()

def plt_data_size_vs_auroc():
    path = os.path.join(os.getcwd(), 'results', 'experiment.json')
    exp_data = dict()
    auroc_results = dict()
    columns = ['data_size', 'max_auroc']
    auroc_results = {label: [] for label in columns}
    if os.path.exists(path):
        with open(path) as json_file:
            exp_data = json.load(json_file)
    for exp_id in exp_data.keys():
        if not exp_data[exp_id]["results"]:
            continue
        if not exp_data[exp_id]["results"]["conducted"]:
            continue
        _size = exp_data[exp_id]["settings"]['traj_length'] / exp_data[exp_id]["settings"]['sample_freq']
        _size = int(_size * exp_data[exp_id]["settings"]['num_sim'])
        auroc_avg = exp_data[exp_id]["results"]['auroc']['max']
        auroc_results['data_size'].append(_size)
        auroc_results['max_auroc'].append(auroc_avg)
        print(auroc_results['data_size'])
    df = pd.DataFrame(auroc_results)
    print(df)
    p = sns.lineplot(df.data_size, df.max_auroc)
    fig = p.get_figure()
    fig.savefig('/Users/naveenmysore/Documents/tau_auroc.png')
    #plt.xscale('log')
    plt.show()

#plt_data_size_vs_auroc()


def traj_snapshot(time_step=0):
    # observations -> positions
    # springs -> spring constants
    # causal graph from predictions

    path = os.path.join(os.getcwd(), 'meta.json')
    meta_data = dict()
    if os.path.exists(path):
        with open(path) as json_file:
            meta_data = json.load(json_file)
    exp_id = meta_data['recent']

    experiment = Experiment()
    experiment.set_id(id=exp_id)
    experiment.load_settings()
    _observations = pd.read_csv(f'/Users/naveenmysore/Documents/data/csdi_data/observations_{exp_id}.csv')
    _springs = pd.read_csv(f'/Users/naveenmysore/Documents/data/csdi_data/springs_{exp_id}.csv')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # ----- Plotting Particle positions
    axes[0].set_title('Particle position')
    entries = []
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
                         ax=axes[0])
    pl.set_ylim(-5.0, 5.0)
    pl.set_xlim(-5.0, 5.0)

    # ----- Plotting spring constants
    axes[0].set_title(f'Spring connections')
    columns = [f'particle_{i}' for i in range(experiment.get_numb_of_particles())]
    s_mat = np.zeros(shape=(experiment.get_numb_of_particles(), experiment.get_numb_of_particles()))
    npr = experiment.get_numb_of_particles()
    for p_a in range(npr):
        for p_b in range(npr):
            if p_a != p_b:
                s_mat[p_a][p_b] = _springs.iloc[time_step][f's_{p_a}_{p_b}']
    s_mat = np.reshape(s_mat, (npr, npr))
    sns.heatmap(pd.DataFrame(s_mat, columns=columns, index=columns),
                vmin=0.0, vmax=2.0, ax=axes[1])

    fig.suptitle(f'Time step {time_step}')

    #plt.show()
    fig.savefig(os.path.join(os.getcwd(), 'tmp', f'graph_{time_step}.png'))
    plt.clf()
    plt.close(fig)


def plot_trajectory():
    import os
    from PIL import Image

    path = os.path.join(os.getcwd(), 'meta.json')
    meta_data = dict()
    if os.path.exists(path):
        with open(path) as json_file:
            meta_data = json.load(json_file)
    exp_id = meta_data['recent']

    experiment = Experiment()
    experiment.set_id(id=exp_id)
    experiment.load_settings()

    time_steps = range(args.length)
    with Pool(4) as p:
        p.map(traj_snapshot, time_steps)


    # Merge all plots
    fcont = len(glob.glob(f"{os.getcwd()}/tmp/graph_*.png"))
    # ref: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f"{os.getcwd()}/tmp/graph_{i}.png") for i in range(1, fcont)]
    img.save(fp=f"{os.getcwd()}/media/trajectory.gif",
             format='GIF',
             append_images=imgs,
             save_all=True,
             duration=10,
             loop=0)

    # delete all png files.
    fp_in = f"{os.getcwd()}/tmp/graph_*.png"
    for f in glob.glob(fp_in):
        os.remove(f)

    return


def main():
    if args.op == 'trajectory':
        print('Plotting trajectory')
        plot_trajectory()
    else:
        plt_data_size_vs_auroc()
    print('Done.')


if __name__ == "__main__":
    main()

