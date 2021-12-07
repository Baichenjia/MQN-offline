import gym
from torch.distributions import Bernoulli
import h5py
import torch
import numpy as np

# 定义 reward wrapper, 供同目录下的 get_env 中的函数调用

__all__ = ['RewardHighVelocity',
           'RewardUnhealthyPose',
           'RewardScale']


class RewardHighVelocity(gym.RewardWrapper):
    """Wrapper to modify environment rewards of 'Cheetah','Walker' and
    'Hopper'.

    Penalizes with certain probability if velocity of the agent is greater
    than a predefined max velocity.
    Parameters
    ----------
    kwargs: dict
    with keys:
    'prob_vel_penal': prob of penalization
    'cost_vel': cost of penalization
    'max_vel': max velocity

    Methods
    -------
    step(action): next_state, reward, done, info
    execute a step in the environment.
    """

    def __init__(self, env, **kwargs):
        super(RewardHighVelocity, self).__init__(env)
        self.penal_v_distr = Bernoulli(kwargs['prob_vel_penal'])
        self.penal = kwargs['cost_vel']
        self.max_vel = kwargs['max_vel']
        allowed_envs = ['Cheetah', 'Hopper', 'Walker']
        assert(any(e in self.env.unwrapped.spec.id for e in allowed_envs)), \
            'Env {self.env.unwrapped.spec.id} not allowed for RewardWrapper'

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        vel = info['x_velocity']
        info['risky_state'] = vel > self.max_vel
        info['angle'] = self.env.sim.data.qpos[2]

        if 'Cheetah' in self.env.unwrapped.spec.id:
            return (observation, self.new_reward(reward, info),
                    done, info)
        if 'Walker' in self.env.unwrapped.spec.id:
            return (observation, self.new_reward(reward, info),
                    done, info)
        if 'Hopper' in self.env.unwrapped.spec.id:
            return (observation, self.new_reward(reward, info),
                    done, info)

    def new_reward(self, reward, info):
        if 'Cheetah' in self.env.unwrapped.spec.id:
            forward_reward = info['reward_run']
        else:
            forward_reward = info['x_velocity']

        penal = info['risky_state'] * \
            self.penal_v_distr.sample().item() * self.penal

        # If penalty applied, substract the forward_reward from total_reward
        # original_reward = rew_healthy + forward_reward - cntrl_cost
        new_reward = penal + reward + (penal != 0) * (-forward_reward)
        return new_reward

    @property
    def name(self):
        return f'{self.__class__.__name__}{self.env}'


class RewardUnhealthyPose(gym.RewardWrapper):
    """Wrapper to modify environment rewards of 'Walker' and 'Hopper'.
    Penalizes with certain probability if pose of the agent doesn't lie
    in a 'robust' state space.
    Parameters
    ----------
    kwargs: dict
    with keys:
    'prob_pose_penal': prob of penalization
    'cost_pose': cost of penalization

    Methods
    -------
    step(action): next_state, reward, done, info
    execute a step in the environment.
    """

    def __init__(self, env, **kwargs):

        super(RewardUnhealthyPose, self).__init__(env)

        self.penal_distr = Bernoulli(kwargs['prob_pose_penal'])
        self.penal = kwargs['cost_pose']
        if 'Walker' in self.env.unwrapped.spec.id:
            self.robust_angle_range = (-0.5, 0.5)
            self.healthy_angle_range = (-1, 1)  # default env

        elif 'Hopper' in self.env.unwrapped.spec.id:
            self.robust_angle_range = (-0.1, 0.1)
            self.healthy_angle_range = (-0.2, 0.2)  # default env

        else:
            raise ValueError('Environment is not Walker neither Hopper '
                             f'for {self.__class__.__name__}')

    @property
    def is_robust_healthy(self):
        z, angle = self.env.sim.data.qpos[1:3]
        min_angle, max_angle = self.robust_angle_range
        robust_angle = min_angle < angle < max_angle
        is_robust_healthy = robust_angle  # and healthy_z
        return is_robust_healthy

    @property
    def is_healthy(self):
        z, angle = self.env.sim.data.qpos[1:3]
        h_min_angle, h_max_angle = self.healthy_angle_range
        healthy_angle = h_min_angle < angle < h_max_angle
        self.is_healthy = healthy_angle

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info['risky_state'] = ~self.is_robust_healthy
        info['angle'] = self.env.sim.data.qpos[2]
        return observation, self.new_reward(reward), done, info

    def new_reward(self, reward):
        # Compute new reward according to penalty probability and agent state:

        # Penalty occurs if agent's pose is not robust with certain prob
        # If env.terminate when unhealthy=False (i.e. episode doesn't finish
        # when unhealthy pose), we do not add penalization when not in
        # healty pose.

        penal = (~self.is_robust_healthy) * (self.is_healthy) *\
            self.penal_distr.sample().item() * self.penal

        new_reward = penal + reward
        return new_reward

    @property
    def name(self):
        return f'{self.__class__.__name__}{self.env}'


class RewardScale(gym.RewardWrapper):
    def __init__(self, env, scale):

        gym.RewardWrapper.__init__(self, env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


class DatasetWriter(object):
    def __init__(self, mujoco=False, goal=False):
        self.mujoco = mujoco
        self.goal = goal
        self.data = self._reset_data()
        self._num_samples = 0

    def _reset_data(self):
        data = {'observations': [],
                'actions': [],
                'terminals': [],
                'rewards': [],
                }
        if self.mujoco:
            data['infos/qpos'] = []
            data['infos/qvel'] = []
            data['infos/reward_ctrl'] = []
            data['infos/reward_run'] = []
            data['infos/x_vel'] = []
            data['infos/angle'] = []
        if self.goal:
            data['infos/goal'] = []
        return data

    def __len__(self):
        return self._num_samples

    def append_data(self, s, a, r, done, goal=None, mujoco_env_data=None,
                    info=None):
        self._num_samples += 1
        self.data['observations'].append(s)
        self.data['actions'].append(a)
        self.data['rewards'].append(r)
        self.data['terminals'].append(done)
        if self.goal:
            self.data['infos/goal'].append(goal)
        if self.mujoco and mujoco_env_data is not None:
            self.data['infos/qpos'].append(mujoco_env_data[0])  # qpos
            self.data['infos/qvel'].append(mujoco_env_data[1])  # qvel
            self.data['infos/x_vel'].append(info['x_velocity'])
            try:
                self.data['infos/angle'].append(info['angle'])
            except KeyError:
                pass

    def write_dataset(self, fname, max_size=None, compression='gzip'):
        np_data = {}
        for k in self.data:
            if k == 'terminals':
                dtype = np.bool_
            else:
                dtype = np.float32
            data = np.array(self.data[k], dtype=dtype)
            if max_size is not None:
                data = data[:max_size]
            np_data[k] = data

        dataset = h5py.File(fname, 'w')
        for k in np_data:
            dataset.create_dataset(k, data=np_data[k], compression=compression)
        dataset.close()


def get_gym_name(dataset_name):
    if 'cheetah' in dataset_name:
        return 'HalfCheetah-v3'
    elif 'hopper' in dataset_name:
        return 'Hopper-v3'
    elif 'walker' in dataset_name:
        return 'Walker2d-v3'
    else:
        raise ValueError("{dataset_name} is not in D4RL")


class HDF5_Creator:

    """Create a hdf5 file containing training data for offline RL algorithm
    with a new reward function.
    Original dataset is obtained from D4RL environment but new data
    is obtained by running version 3 of the same environment type to get
    additional information only provided in version 3.

    Parameters
    ----------
    d4rl_env_name: str
                Name of the original hdf5 provided by D4RL.
                Example: 'walker2d-expert-v0'
    properties_env: dict
            dictionary containing information about the new reward function
            For speed penalization:  dict_env = {'prob_vel_penal': float, 'cost_vel': int, 'max_vel': float}
            For pose penalization:  dict_env = {'prob_pose_penal':float, 'cost_pose': int,}
    """

    def __init__(self, d4rl_env_name, properties_env=None, fname=None):
        self.env_d4rl = gym.make(d4rl_env_name)
        self.d4rl_env_name = d4rl_env_name
        self.dataset = self.env_d4rl.get_dataset()
        self.writer = DatasetWriter(mujoco=True)    # 自己定义的类, 类里面包含很多list, 用来存储 s,a,r,s' 等

        env = gym.make(get_gym_name(d4rl_env_name)).unwrapped   # make 一个正常的 mujoco 环境, 如 "Walker2d-v3". 不是 offline 环境
        env.seed(10)
        torch.manual_seed(10)
        np.random.seed(10)
        self.actions = self.dataset['actions']    # (1000000, 6) 对于 Walker2d-v3 环境
        self.obs = self.dataset['observations']   # (1000000, 17)
        self.rewards = self.dataset['rewards']    # (1000000,)
        self.dones = self.dataset['terminals']    # (1000000,)
        self.properties_env = properties_env      # {'name': 'walker2d-expert-v0', 'prob_pose_penal': 0.15, 'cost_pose': -30}
        print("***", get_gym_name(d4rl_env_name), self.actions.shape, self.obs.shape, self.rewards.shape, self.dones.shape, self.properties_env)

        # 根据 properties_env 的参数构建文件名, 数据集构建好之后将存储在 ~/.d4rl/dataset 目录下
        dataset_name = self.env_d4rl.dataset_filepath[:-5]

        if properties_env.get('cost_vel', False):
            self.env = RewardHighVelocity(env, **properties_env)
            self.h5py_name = f'{dataset_name}_'\
                f'prob{properties_env["prob_vel_penal"]}_'\
                f'penal{properties_env["cost_vel"]}_'\
                f'maxvel{properties_env["max_vel"]}.hdf5'

        elif properties_env.get('cost_pose', False):
            self.env = RewardUnhealthyPose(env, **properties_env)
            self.h5py_name = f'{dataset_name}_'\
                f'prob{properties_env["prob_pose_penal"]}_'\
                f'penal{properties_env["cost_pose"]}_'\
                'pose.hdf5'

        else:
            raise ValueError('No reward wrapper found')

        # dataset_name = self.env_d4rl.dataset_filepath[:-5]
        # fname 是一个绝对路径
        assert fname == self.h5py_name, \
            f'Not same name for h5py file {fname} vs {self.h5py_name}'

    def create_hdf5_file(self):
        if 'cheetah' in self.d4rl_env_name:
            # Need to rerun environment since vel is not provided in obs:
            self.create_hdf5_file_cheetah()
        else:
            # Apply reward function on the reward
            self.create_hdf5_file_hopper_walker()
        self.check_data()

    def create_hdf5_file_hopper_walker(self):
        # 专用于对 walker 和 hopper 来生成数据集
        print('\n\n **** Creating new dataset...hopper-walker******\n\n')
        min_angle, max_angle = self.env.robust_angle_range
        penal_distr = Bernoulli(self.properties_env['prob_pose_penal'])

        for i in range(len(self.actions)):  # 循环 dataset 中的每一个元素
            observation = self.obs[i]       # 对于当前的 obs, 根据约束定义一些 penal, 作为 reward
            # differs from env.sim.data.qpos[1:3] in RewardWrapper
            # since in observation xposition (qpos[0]) has already been excluded
            _, angle = observation[0], observation[1]
            robust_angle = min_angle < angle < max_angle
            penal = (~robust_angle) *\
                penal_distr.sample().item() * self.properties_env['cost_pose']
            r = penal + self.rewards[i]
            if not i % 10000:
                print(f'Num datapoint {i}/{len(self.actions)}')
            self.writer.append_data(observation, self.actions[i], r, self.dones[i])

        self.writer.write_dataset(fname=self.h5py_name)

    def get_state(self, i):
        pos_full = np.concatenate([[self.env.sim.data.qpos[0].copy()], self.obs[i][0:8]])  # pos_full = [xpos, 8jointpos]
        vel_full = self.obs[i][8::]  # [9 jointvels]
        return pos_full, vel_full

    def create_hdf5_file_cheetah(self):
        # 专用于对 cheetah 来生成数据集
        self.writer._reset_data()
        init_pos = np.concatenate([[0], self.obs[0][0:8]])  # 5 for hopper
        init_vel = self.obs[0][8::]

        self.env.reset()
        print(len(init_pos), (len(init_vel)))  # check
        self.env.set_state(qpos=init_pos, qvel=init_vel)

        print('\n\n **** Creating new dataset...cheetah******\n\n')
        for i in range(len(self.actions)):
            pos_full, vel = self.get_state(i)  # contains xposition
            observation = np.concatenate(
                (pos_full[1:], vel)).ravel()  # remove xposition
            action = self.actions[i]
            if not i % 10000:
                print(f'Num datapoint {i}/{len(self.actions)}')

            # reset state to data, needed to run open-loop
            self.env.set_state(qpos=pos_full, qvel=vel)
            _, reward, _, info = self.env.step(self.actions[i])

            if self.properties_env is None:  # no probpenal
                r = self.rewards[i]
            else:
                r = reward
            self.writer.append_data(observation, action, r, self.dones[i],
                                    mujoco_env_data=[pos_full, vel], info=info)

        self.writer.write_dataset(fname=self.h5py_name)

    def check_data(self):
        print(f'Checking dataset {self.h5py_name} is correct...')
        # Use get_dataset method from OfflineEnv Class in
        # D4RL to check dataset
        new_dataset = self.env_d4rl.get_dataset(h5path=self.h5py_name)

        for _ in range(10):
            random_datapoint = np.random.randint(0, len(self.dataset))
            assert all(self.dataset['actions'][random_datapoint] ==
                       new_dataset['actions'][random_datapoint]),\
                f'{self.h5py_name} is not correct. Not same actions!'
            assert all(self.dataset['observations'][random_datapoint] ==
                       new_dataset['observations'][random_datapoint]),\
                f'{self.h5py_name} is not correct. Not same observations!'

        print(f'Dataset correct. Dataset saved in {self.h5py_name}')
