import d4rl
import gym
import numpy as np
import torch
import os
import h5py

from torch.utils.data import TensorDataset, DataLoader
from reward_wrappers import RewardHighVelocity, RewardUnhealthyPose, HDF5_Creator


def combine_d4rl_dataset(env_name='hopper', threshold=250000):
	data_types = ['random', 'medium', 'medium-replay', 'expert']

	dataset = None
	for data_type in data_types:
		dataset_name = f'{env_name}-{data_type}-v0'
		env, current_dataset = load_d4rl_dataset(dataset_name)
		if dataset is None:
			dataset = current_dataset
			for key in dataset.keys():
				dataset[key] = dataset[key][:threshold]
		else:
			for key in dataset.keys():
				dataset[key] = np.concatenate([dataset[key], current_dataset[key][:threshold]], axis=0)

	return env, dataset


def load_d4rl_dataset(env_name='halfcheetah-expert-v0'):
	env = gym.make(env_name)
	dataset = d4rl.qlearning_dataset(env)
	# dataset['rewards'] = np.expand_dims(dataset['rewards'], axis=1)
	# dataset['terminals'] = np.expand_dims(dataset['terminals'], axis=1)
	return env, dataset


def load_normalized_dataset(env_name='hopper', dataset_name='medium-replay-v0'):
	x_train, y_train, x_test, y_test = np.load(f'data/{env_name}-{dataset_name}-normalized-data.npy', allow_pickle=True)
	return x_train, y_train, x_test, y_test


def multistep_dataset(env, h=2, terminate_on_end=False, **kwargs):
	dataset = env.get_dataset(**kwargs)
	N = dataset['rewards'].shape[0]

	obs_ = []
	next_obs_ = []
	action_ = []
	reward_ = []
	done_ = []

	use_timeouts = False
	if 'timeouts' in dataset:
		use_timeouts = True

	episode_step = 0
	for i in range(N - h):
		skip = False
		for j in range(i, i + h - 1):
			if bool(dataset['terminals'][j]) or dataset['timeouts'][j]:
				skip = True
		if skip:
			continue

		obs = dataset['observations'][i]
		new_obs = dataset['observations'][i + h]
		action = dataset['actions'][i:i + h].flatten()
		reward = dataset['rewards'][i + h - 1]
		done_bool = bool(dataset['terminals'][i + h - 1])

		if use_timeouts:
			final_timestep = dataset['timeouts'][i + h - 1]
		else:
			final_timestep = (episode_step == env._max_episode_steps - 1)
		if (not terminate_on_end) and final_timestep:
			# Skip this transition and don't apply terminals on the last step of an episode
			episode_step = 0
			continue
		if done_bool or final_timestep:
			episode_step = 0

		obs_.append(obs)
		next_obs_.append(new_obs)
		action_.append(action)
		reward_.append(reward)
		done_.append(done_bool)
		episode_step += 1

	return {
		'observations': np.array(obs_),
		'actions': np.array(action_),
		'next_observations': np.array(next_obs_),
		'rewards': np.array(reward_),
		'terminals': np.array(done_),
	}


def format_samples_for_training(samples):
	obs = samples['observations']
	act = samples['actions']
	next_obs = samples['next_observations']
	rew = samples['rewards']
	delta_obs = next_obs - obs
	inputs = np.concatenate((obs, act), axis=-1)
	outputs = np.concatenate((rew.reshape(rew.shape[0], -1), delta_obs), axis=-1)

	# inputs = torch.from_numpy(inputs).float()
	# outputs = torch.from_numpy(outputs).float()

	return inputs, outputs


def create_data_loader(X, y, train_n=5000, test_n=6000, batch_size=64):
	train_x, train_y = X[:train_n], y[:train_n]
	test_x, test_y = X[train_n:test_n], y[train_n:test_n]

	train_dataset = TensorDataset(train_x, train_y)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	test_dataset = TensorDataset(test_x, test_y)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, test_loader


def shuffle_rows(arr):
	idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
	return arr[np.arange(arr.shape[0])[:, None], idxs]


def batch_generator(index_array, batch_size):
	index_array = shuffle_rows(index_array)
	batch_count = 0
	while True:
		if batch_count * batch_size + batch_size >= index_array.shape[1]:
			batch_count = 0
			index_array = shuffle_rows(index_array)
		start = batch_count * batch_size
		end = start + batch_size
		batch_count += 1
		yield index_array[:, start:end]


################################################################
# For Risk D4RL dataset. From https://github.com/nuria95/O-RAAC
################################################################

risk_para = {
	"halfcheetah-expert-v0": {
		"prob_vel_penal": 0.05,
		"max_vel": 10,
		"cost_vel": -60
	},
	'halfcheetah-medium-v0': {
		"prob_vel_penal": 0.05,
		"max_vel": 4,
		"cost_vel": -70
	},
	'halfcheetah-medium-replay-v0': {
		"prob_vel_penal": 0.05,
		"max_vel": 4,
		"cost_vel": -70
	},
	"hopper-expert-v0": {
		"prob_vel_penal": None,
		"prob_pose_penal": 0.1,
		"cost_pose": -50
	},
	"hopper-medium-v0": {
		"prob_vel_penal": None,
		"prob_pose_penal": 0.1,
		"cost_pose": -50
	},
	"hopper-medium-replay-v0": {
		"prob_vel_penal": None,
		"prob_pose_penal": 0.1,
		"cost_pose": -50
	},
	"walker2d-expert-v0": {
		"prob_vel_penal": None,
		"prob_pose_penal": 0.1,
		"cost_pose": -30,
		"terminate_when_unhealthy": 1
	},
	"walker2d-medium-v0": {
		"prob_vel_penal": None,
		"prob_pose_penal": 0.1,
		"cost_pose": -30,
		"terminate_when_unhealthy": 1
	},
	"walker2d-medium-replay-v0": {
		"prob_vel_penal": None,
		"prob_pose_penal": 0.1,
		"cost_pose": -30,
		"terminate_when_unhealthy": 1
	}
}


def get_gym_name(dataset_name):
	# Get v3 version of environments for extra information to be available
	if 'cheetah' in dataset_name:
		return 'HalfCheetah-v3'
	elif 'hopper' in dataset_name:
		return 'Hopper-v3'
	elif 'walker' in dataset_name:
		return 'Walker2d-v3'
	else:
		raise ValueError("{dataset_name} is not in D4RL")


path_to_datasets = os.environ.get('D4RL_DATASET_DIR', os.path.expanduser('~/.d4rl/datasets'))


def get_keys(h5file):
	keys = []

	def visitor(name, item):
		if isinstance(item, h5py.Dataset):
			keys.append(name)

	h5file.visititems(visitor)
	return keys


def get_env_risk(env_name, reset_noise_scale=None, eval_terminate_when_unhealthy=True, only_env=False):
	# 1. ?????? offline ????????? make env. ?????? seed
	terminate_when_unhealthy = False \
		if not eval_terminate_when_unhealthy else True

	env_d4rl = gym.make(env_name)  # ?????? make ?????? offline ?????????, ??? walker2d-medium-v0
	dataset_name = env_d4rl.dataset_filepath[:-5]  #

	# Use v3 version of environments for extra information to be available
	kwargs = {'terminate_when_unhealthy': terminate_when_unhealthy} if 'cheetah' not in dataset_name else {}
	if reset_noise_scale is not None:
		kwargs['reset_noise_scale'] = reset_noise_scale

	# 2. ?????? online ?????????
	env = gym.make(get_gym_name(dataset_name), **kwargs).unwrapped  # ???????????? online ??? env, ?????? <Walker2dEnv<Walker2d-v3>>
	env.unwrapped.seed(seed=None)

	# 3. ?????????????????????????????????. ?????? p.env ?????????, ??? p.env ??????????????? json_params ???????????????????????? env ???????????????
	# ?????? prob_vel_penal (????????? json ????????????) ?????? none, ??????????????????????????????
	dict_env = None
	if risk_para[env_name]["prob_vel_penal"] is not None and risk_para[env_name]["prob_vel_penal"] > 0:
		print("Risk Env 1.")
		dict_env = {'prob_vel_penal': risk_para[env_name]["prob_vel_penal"],
					'cost_vel': risk_para[env_name]["cost_vel"],
					'max_vel': risk_para[env_name]["max_vel"]}
		print(dict_env)
		fname = f'{dataset_name}_' \
				f'prob{dict_env["prob_vel_penal"]}_' \
				f'penal{dict_env["cost_vel"]}_' \
				f'maxvel{dict_env["max_vel"]}.hdf5'

		env = RewardHighVelocity(env, **dict_env)  # ?????? reward ?????????
	# ?????? prob_pose_penal (????????? json ????????????) ?????? none, ??????????????????????????????
	elif risk_para[env_name]["prob_pose_penal"] is not None and risk_para[env_name]["prob_pose_penal"] > 0:
		print("Risk Env 2.")
		dict_env = {'prob_pose_penal': risk_para[env_name]["prob_pose_penal"],
					'cost_pose': risk_para[env_name]["cost_pose"]}
		print(dict_env)
		fname = f'{dataset_name}_' \
				f'prob{dict_env["prob_pose_penal"]}_' \
				f'penal{dict_env["cost_pose"]}_' \
				'pose.hdf5'
		env = RewardUnhealthyPose(env, **dict_env)
	else:  # ?????? json ?????????????????????????????????, ?????????????????? d4rl ?????????????????? offline data
		print(" init  ??????????????? offline dataset, ???????????????")
		fname = env_d4rl.dataset_filepath

	if only_env:
		return env

	print("dict_env:\n", dict_env)

	# 3. ??? hdf5 ??? offline ????????????????????????.
	# ???????????????????????????????????????????????? state,action,done ????????????. ?????? reward ??????????????????, ?????? penal.
	h5py_path = os.path.join(path_to_datasets, fname)
	print("h5py_path:", h5py_path)
	if not os.path.exists(h5py_path):
		print(f'\n{h5py_path} doesn\'t exist.')
		creator = HDF5_Creator(d4rl_env_name=env_name, properties_env=dict_env, fname=fname)
		creator.create_hdf5_file()
		creator.check_data()
	else:
		print(f'\nChecking dataset {h5py_path} is correct...')
		env_d4rl.get_dataset(h5path=h5py_path)
		print('Dataset correct\n')

	dataset_file = h5py.File(h5py_path, 'r')
	data = {k: dataset_file[k][:] for k in get_keys(dataset_file)}
	dataset_file.close()

	# load specific dataset
	env = gym.make(env_name)
	dataset = d4rl.qlearning_dataset(env, dataset=data)
	return env, dataset


if __name__ == '__main__':
	# combine_d4rl_dataset('hopper')
	get_env_risk("halfcheetah-expert-v0")
