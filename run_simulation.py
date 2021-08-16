#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
Simulates a chosen system
(Spring Particle, Charge Particle or Gravity Particles)
writes observational data and schema to /data
"""
from simulations.synthetic_sim import ChargedParticlesSim, SpringSim
import time
import numpy as np
import argparse
from experiment import Experiment

parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='springs',
                    help='What simulation to generate.')
parser.add_argument('--num-train', type=int, default=10,
                    help='Number of training simulations to generate.')
parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--length-test', type=int, default=10000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n-balls', type=int, default=5,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--min_change_step', type=int, default=1000, help='minimum step of changing interaction')
parser.add_argument('--max_change_step', type=int, default=1000, help='maximum step of changing interaction')
args = parser.parse_args()

if args.simulation == 'springs':
    sim = SpringSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = '_springs'
elif args.simulation == 'charged':
    sim = ChargedParticlesSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = '_charged'
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

suffix += '_{}_min{}_max{}'.format(str(args.n_balls),args.min_change_step,args.max_change_step)

np.random.seed(args.seed)

print(suffix)

experiment = Experiment()

# *** Control Variables ***
experiment.set_numb_of_particles(num_of_particles=args.n_balls)
experiment.set_traj_length(traj_length=args.length)
experiment.set_sample_freq(sample_freq=args.sample_freq)
experiment.set_period(period=args.max_change_step)
experiment.set_num_sim(num_sim=args.num_train)
# experiment.set_initial_velocity(vel=args.vel)
# ********

particle_observations = experiment.get_particle_observational_record()
spring_observations = experiment.get_springs_observational_record()


def generate_dataset(num_sims, length, sample_freq):
    loc_all = list()
    vel_all = list()
    edges_all = list()

    for i in range(num_sims):
        t = time.time()
        loc, vel, edges = sim.sample_trajectory_dynamic(T=length,
                                                        sample_freq=sample_freq,
                                                        min_step=args.min_change_step,
                                                        max_step=args.max_change_step)

        for ind, lc in enumerate(loc):
            x_positions, y_positions = lc[0], lc[1]
            x_vel, y_vel = vel[ind][0], vel[ind][1]
            observation = dict()
            observation[f'trajectory_step'] = f'{i}_{ind}'
            for particle_id in range(args.n_balls):
                observation[f'p_{particle_id}_x_position'] = x_positions[particle_id]
                observation[f'p_{particle_id}_y_position'] = y_positions[particle_id]
                observation[f'p_{particle_id}_x_velocity'] = x_vel[particle_id]
                observation[f'p_{particle_id}_y_velocity'] = y_vel[particle_id]

            sp_observation = dict()
            sp_observation[f'trajectory_step'] = f'{i}_{ind}'
            for si in range(args.n_balls):
                for sj in range(args.n_balls):
                    if si != sj:
                        sp_observation[f's_{si}_{sj}'] = edges[ind][si][sj]

            particle_observations.add_an_observation(observation)
            spring_observations.add_an_observation(sp_observation)

        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)
        vel_all.append(vel)
        edges_all.append(edges)


print('Running simulations..')
generate_dataset(args.num_train, args.length, args.sample_freq)
particle_observations.save_observations(path='/Users/naveenmysore/Documents/data/csdi_data/',
                                        name=f'observations_{experiment.get_id()}')
spring_observations.save_observations(path='/Users/naveenmysore/Documents/data/csdi_data/',
                                      name=f'springs_{experiment.get_id()}')
experiment.save()
print('Saved.')
