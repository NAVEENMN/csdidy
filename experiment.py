# !/usr/bin/env python3.8
# -*- coding: utf-8 -*-
import os
import json
import logging
import pandas as pd
import datetime
import pytz

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


timezone = pytz.timezone("America/Los_Angeles")
dt = timezone.localize(datetime.datetime.now())


class Observations(object):
    def __init__(self):
        self.column_names = []
        self.observations = dict()

    def set_column_names(self, columns):
        self.column_names = columns
        self.observations = {label: [] for label in columns}

    def add_an_observation(self, observation):
        for col_name in observation.keys():
            self.observations[col_name].append(observation[col_name])

    def get_observations(self):
        df = pd.DataFrame(self.observations)
        return df

    def save_observations(self, path, name):
        logging.info("*** Saving: observations")
        df = pd.DataFrame(self.observations).set_index('trajectory_step')
        # index represents most reset observation
        df.to_csv(os.path.join(path, f'{name}.csv'))
        logging.info(f"*** Saved: observations {name}.csv")


class Experiment(object):
    def __init__(self):
        self._id = self._get_experiment_id()
        self.num_of_particles = 0
        self.tau_min = 1
        self.tau_max = 10
        self.traj_length = 0
        self.sample_freq = 0
        self.num_sim = 0
        self.period = 0
        self.init_vel = 0.0
        self.particle_vars = []
        self.spring_vars = []
        self.p_threshold = 0.05

    def get_id(self):
        return self._id

    def set_id(self, id):
        self._id = id

    def load_settings(self):
        path = os.path.join(os.getcwd(), 'experiment.json')
        _id = self.get_id()
        exp_data = {}
        if os.path.exists(path):
            with open(path) as json_file:
                exp_data = json.load(json_file)
        self.set_tau_min(exp_data[self._id]['settings']['tau_min'])
        self.set_tau_max(exp_data[self._id]['settings']['tau_max'])
        self.set_traj_length(exp_data[self._id]['settings']['traj_length'])
        self.set_num_sim(exp_data[self._id]['settings']['num_sim'])
        self.set_sample_freq(exp_data[self._id]['settings']['sample_freq'])
        self.set_period(exp_data[self._id]['settings']['period'])
        self.set_initial_velocity(exp_data[self._id]['settings']['initial_velocity'])
        self.set_numb_of_particles(exp_data[self._id]['settings']['number_of_particles'])
        self.particle_vars = exp_data[self._id]['variables']['particles']
        self.spring_vars = exp_data[self._id]['variables']['springs']

    def load_results(self):
        path = os.path.join(os.getcwd(), 'experiment.json')
        _id = self.get_id()
        exp_data = {}
        if os.path.exists(path):
            with open(path) as json_file:
                exp_data = json.load(json_file)
        self.set_p_threshold(exp_data[self._id]['results']['p_threshold'])

    def get_vars(self):
        return self.particle_vars, self.spring_vars

    def set_tau_min(self, tau_min):
        self.tau_min = tau_min

    def get_tau_min(self):
        return self.tau_min

    def set_tau_max(self, tau_max):
        self.tau_max = tau_max

    def get_tau_max(self):
        return self.tau_max

    def set_traj_length(self, traj_length):
        self.traj_length = traj_length

    def get_traj_length(self):
        return self.traj_length

    def set_sample_freq(self, sample_freq):
        self.sample_freq = sample_freq

    def get_sample_freq(self):
        return self.sample_freq

    def set_num_sim(self, num_sim):
        self.num_sim = num_sim

    def get_num_of_sim(self):
        return self.num_sim

    def set_numb_of_particles(self, num_of_particles):
        self.num_of_particles = num_of_particles

    def get_numb_of_particles(self):
        return self.num_of_particles

    def set_period(self, period):
        self.period = period

    def get_period(self):
        return self.period

    def set_initial_velocity(self, vel):
        self.init_vel = vel

    def get_initial_velocity(self):
        return self.init_vel

    def set_p_threshold(self, p_th):
        self.p_threshold = p_th

    def get_p_threshold(self):
        return self.p_threshold

    def _get_particle_vars(self):
        np = self.get_numb_of_particles()
        column_names = []
        column_names.extend([f'p_{particle_id}_x_position' for particle_id in range(np)])
        column_names.extend([f'p_{particle_id}_y_position' for particle_id in range(np)])
        column_names.extend([f'p_{particle_id}_x_velocity' for particle_id in range(np)])
        column_names.extend([f'p_{particle_id}_y_velocity' for particle_id in range(np)])
        return column_names

    def get_particle_observational_record(self):
        particle_observations = Observations()
        _vars = self._get_particle_vars()
        _vars.append('trajectory_step')
        particle_observations.set_column_names(columns=_vars)
        return particle_observations

    def _get_springs_vars(self):
        np = self.get_numb_of_particles()
        column_names = []
        for i in range(np):
            for j in range(np):
                if i != j:
                    column_names.append(f's_{i}_{j}')
        return column_names

    def get_springs_observational_record(self):
        spring_observations = Observations()
        _vars = self._get_springs_vars()
        _vars.append('trajectory_step')
        spring_observations.set_column_names(columns=_vars)
        return spring_observations

    def fmt_id(self, _id):
        #_time = f'{dt.time().hour}:{dt.time().minute}:{dt.time().second}:{dt.time().microsecond}'
        _day = f'{dt.date().month}:{dt.date().day}:{dt.date().year}'
        name = f'exp_{_day}-{_id}'
        return name

    def _get_experiment_id(self):
        path = os.path.join(os.getcwd(), 'experiment.json')
        _id = self.fmt_id(_id=1)
        if os.path.exists(path):
            with open(path) as json_file:
                exp_data = json.load(json_file)
                _id = self.fmt_id(_id=len(exp_data.keys())+1)
        return _id

    def save(self):
        logging.info(f'*** Utils: Saving experiment {self._id}')
        path = os.path.join(os.getcwd(), 'experiment.json')
        exp_data = dict()
        if os.path.exists(path):
            with open(path) as json_file:
                exp_data = json.load(json_file)
        exp_data[self._id] = {
            'settings': {
                'tau_min': self.get_tau_min(),
                'tau_max': self.get_tau_max(),
                'traj_length': self.get_traj_length(),
                'sample_freq': self.get_sample_freq(),
                'period': self.get_period(),
                'num_sim': self.get_num_of_sim(),
                'number_of_particles': self.get_numb_of_particles(),
                'initial_velocity': self.get_initial_velocity()
            },
            'variables': {
              'particles': self._get_particle_vars(),
              'springs': self._get_springs_vars()
            },
            'results': {
                'conducted': False,
                'tau': None,
                'p_threshold': None,
                'auroc': {
                    'mean': None,
                    'max': None,
                    'min': None,
                    'std': None
                }
            }
        }
        with open(path, 'w') as f:
            json.dump(exp_data, f, indent=4)
        logging.info(f'*** Utils: Saved experiment {self._id} settings to {path}')

        meta_path = os.path.join(os.getcwd(), 'meta.json')
        meta_data = {}
        if os.path.exists(meta_path):
            with open(meta_path) as j_file:
                meta_data = json.load(j_file)
        meta_data["recent"] = self.get_id()
        meta_data["date"] = f'{dt.date().month}:{dt.date().day}:{dt.date().year}'
        meta_data["time"] = f'{dt.time().hour}:{dt.time().minute}:{dt.time().second}:{dt.time().microsecond}'
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f, indent=4)
        logging.info(f'*** Utils: meta data for experiment {self._id} logged to {meta_path}')