import argparse
import os
from d4rl.infos import REF_MIN_SCORE, REF_MAX_SCORE

import wandb
from sac import SAC, CQL, ReplayMemory
from models import ProbEnsemble, PredictEnv
from batch_utils import *
from mbrl_utils import *
from utils import *
import csv
from tqdm import tqdm
import datetime
import json


MODEL_FREE = ['sac', 'cql', 'codac', 'codac_cvar', 'wcql']


def readParser():
	parser = argparse.ArgumentParser(description='BATCH_RL')
	parser.add_argument('--wandb', type=bool, default=False)
	parser.add_argument('--env', default="halfcheetah-medium-replay-v0", help='Mujoco Gym environment with Risk (default: hopper-medium-v0)')
	parser.add_argument('--algo', default="wcql")
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--version', type=int, default=2, help='CODAC policy update version')
	parser.add_argument('--tau_type', default="iqn")
	parser.add_argument('--dist_penalty_type', default="uniform")
	parser.add_argument('--entropy', default="false")
	parser.add_argument('--lag', type=float, default=10.0)
	parser.add_argument('--min_z_weight', type=float, default=10.0)
	parser.add_argument('--actor_lr', type=float, default=0.00003)
	parser.add_argument('--risk_type', default="cvar")
	parser.add_argument('--risk_param', default=0.1, type=float)
	# risk parameters for the environment
	parser.add_argument('--risk_prob', type=float, default=0.8)
	parser.add_argument('--risk_penalty', type=float, default=200)
	parser.add_argument('--use_bc', dest='use_bc', action='store_true')
	parser.set_defaults(use_bc=False)
	parser.add_argument('--pretrained', dest='pretrained', action='store_true')
	parser.set_defaults(pretrained=False)
	parser.add_argument('--penalty', type=float, default=1.0, help='reward penalty')
	parser.add_argument('--rollout_length', type=int, default=1, metavar='A', help='rollout length')
	parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
	parser.add_argument('--comments', type=str, default='')
	parser.add_argument('--codac_cvar', type=float, default=0.0, help='codac_cvar')
	parser.add_argument('--wcql_cvar', type=float, default=0.0, help='wcql_cvar')

	parser.add_argument('--replay_size', type=int, default=2000000, metavar='N',
						help='size of replay buffer (default: 10000000)')
	parser.add_argument('--model_retain_epochs', type=int, default=5, metavar='A',
						help='retain epochs')
	parser.add_argument('--model_train_freq', type=int, default=1000, metavar='A',
						help='frequency of training')
	parser.add_argument('--rollout_batch_size', type=int, default=50000, metavar='A',
						help='rollout number M')
	parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
						help='steps per epoch')
	parser.add_argument('--num_epoch', type=int, default=1000, metavar='A',
						help='total number of epochs')
	parser.add_argument('--dataset_epoch', type=int, default=100)
	parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
						help='ratio of env samples / model samples')
	parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
						help='initial random exploration steps')
	parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
						help='frequency of training policy')
	parser.add_argument('--num_train_repeat', type=int, default=1, metavar='A',
						help='times to training policy per step')
	parser.add_argument('--eval_n_episodes', type=int, default=10, metavar='A',
						help='number of evaluation episodes')
	parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
						help='max training times per step')
	parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
						help='batch size for training policy')
	parser.add_argument('--model_type', default='pytorch', metavar='A',
						help='predict model -- pytorch or tensorflow')
	parser.add_argument('--pre_trained', type=bool, default=False,
						help='flag for whether dynamics model pre-trained')
	return parser.parse_args()


def get_normalized_score(env_name, reward):
	min_score = REF_MIN_SCORE[env_name]
	max_score = REF_MAX_SCORE[env_name]
	normalized_score = 100 * (reward - min_score) / (max_score - min_score)
	return normalized_score


def evaluate_risk(args, env_sampler, agent, epoch_length=1000):
	env_sampler.current_state = None
	env_sampler.path_length = 0

	sum_reward = 0
	x_position = []
	risky_state = []
	x_velocity = []
	angle = []
	for t in range(epoch_length):
		state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
		x_position.append(info['x_position'])
		risky_state.append(info['risky_state'])
		x_velocity.append(info['x_velocity'])
		angle.append(info['angle'])

		sum_reward += reward
		if done:
			break
	# reset the environment
	env_sampler.current_state = None
	env_sampler.path_length = 0

	return sum_reward, np.array(x_position), np.array(risky_state), np.array(x_velocity), np.array(angle)


def eval_policy(args, env_sampler, predict_env, agent):
	total_step = 0

	rewards, risks, velocity, angles = [], [], [], []
	for i in range(100):
		rewards_episode, x_position, risky_state, x_velocity, angle = \
			evaluate_risk(args, env_sampler, agent, args.epoch_length)

		rewards.append(get_normalized_score(args.env, rewards_episode))
		risks.append(np.mean(risky_state))
		velocity.append(np.mean(x_velocity))
		angles.append(np.mean(angle))
		print(f"episode:{i}, reward:{rewards[-1]}, risk:{risks[-1]}, velocity:{velocity[-1]}, angle: {angles[-1]}")

	rewards = np.array(rewards)
	rewards_avg = np.mean(rewards, axis=0)
	rewards_std = np.std(rewards, axis=0)
	sorted_rewards = np.sort(rewards)
	cvar = sorted_rewards[:int(0.1 * sorted_rewards.shape[0])].mean()

	print("", flush=True)
	print(f'Eval_Reward {rewards_avg:.2f} Eval_Cvar {cvar:.2f} Eval_Std {rewards_std:.2f}', flush=True)
	print(f"Risk:{np.mean(risks)}, Velocity:{np.mean(velocity)}, Angles:{np.mean(angles)}")


def main():
	args = readParser()

	run_name = f"offline-{args.algo}-{args.dist_penalty_type}-{args.risk_type}{args.risk_param}-{args.seed}"
	if 'dsac' in args.algo:
		run_name = f"offline-{args.algo}-{args.dist_penalty_type}-{args.risk_type}{args.risk_param}-Z{args.min_z_weight}-L{args.lag}-E{args.entropy}-{args.seed}"

	# Initial environment
	args.num_epoch = 1001
	args.entropy_tuning = False

	# Load config from config file
	try:
		if 'wcql' in args.algo:
			config = json.load(open('configs/d4rl_configs_wcql.json', 'r'))[args.env]
		else:
			config = json.load(open('configs/d4rl_configs.json', 'r'))[args.env]
		args.min_z_weight = config['min_z_weight']
		args.lag = config['lag']
		args.actor_lr = config['actor_lr']
		args.entropy = config['entropy']
		print("load config:", "min_z_weight:", args.min_z_weight, ", lag:", args.lag,
			", actor_lr:", args.actor_lr, ", entropy:", args.entropy)
	except:
		pass

	# if args.entropy == "true":
	# 	args.entropy_tuning = True
	# args.adapt = False
	# args.d4rl = False

	# 构建 risk 的环境
	assert args.env in ["halfcheetah-medium-v0", "halfcheetah-expert-v0", "hopper-medium-v0",
						"hopper-expert-v0", "walker2d-expert-v0", "walker2d-medium-v0",
						"halfcheetah-medium-replay-v0", "walker2d-medium-replay-v0", "hopper-medium-replay-v0"]

	env = get_env_risk(args.env, only_env=True)
	# print("Load offline data ", os.path.split(env.dataset_url)[-1])
	# args.dataset = args.env
	# args.d4rl = True
	# print("Load risk dataset done.")

	# args.save_dir = f'result-{args.algo}cvar{args.comments}/{args.env}-{datetime.datetime.now().strftime("%m-%d-%H-%M-%S")}/seed-{run_name}'
	# os.makedirs(args.save_dir, exist_ok=True)
	# os.makedirs(f'{args.save_dir}/models', exist_ok=True)

	# with open(f"{args.save_dir}/train.csv", "w") as train_csv:
	# 	train_writer = csv.writer(train_csv)
	# 	train_writer.writerow(["steps", "critic_1_loss", "critic_2_loss", "policy_loss", "ent_loss", "alpha",
	# 						   "Training/z1_pred", "Training/z1_pred_high", "Training/z1_pred_low",
	# 						   "Training/z2_pred", "Training/z2_pred_high", "Training/z2_pred_low",
	# 						   "Training/q1_rand", "Training/q1_curr_actions", "Training/q1_next_actions"])
	# with open(f"{args.save_dir}/eval.csv", "w") as eval_csv:
	# 	eval_writer = csv.writer(eval_csv)
	# 	eval_writer.writerow(["Epochs", "Eval_Reward", "Eval_Cvar", "Eval_Std", "d4rl scores"])

	# only use batch data for model-free methods
	if args.algo in MODEL_FREE:
		args.real_ratio = 1.0

	args.run_name = run_name

	# Set random seed
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	env.seed(args.seed)
	env.action_space.seed(args.seed)

	# gpu
	torch.cuda.set_device(args.gpu)
	device = torch.device("cuda:" + str(args.gpu))

	if args.algo == 'sac':
		agent = SAC(env.observation_space.shape[0], env.action_space,
					automatic_entropy_tuning=args.entropy_tuning)
	elif args.algo == 'codac':
		from distributional.codac import CODAC
		agent = CODAC(env.observation_space.shape[0], env.action_space,
					  version=args.version,
					  tau_type=args.tau_type, use_bc=args.use_bc,
					  min_z_weight=args.min_z_weight, actor_lr=args.actor_lr,
					  risk_type=args.risk_type, risk_param=args.risk_param,
					  dist_penalty_type=args.dist_penalty_type,
					  lagrange_thresh=args.lag,
					  use_automatic_entropy_tuning=args.entropy_tuning,
					  device=device)
	elif args.algo == 'cql':
		from sac.cql import CQL
		args.dist_penalty_type = 'none'
		agent = CQL(env.observation_space.shape[0], env.action_space,
					min_q_weight=args.min_z_weight, policy_lr=args.actor_lr,
					lagrange_thresh=args.lag,
					automatic_entropy_tuning=args.entropy_tuning,
					device=device)
	elif args.algo == 'wcql':
		from distributional.wcql import WCQL
		agent = WCQL(env.observation_space.shape[0], env.action_space,
					 version=args.version,
					 tau_type=args.tau_type, use_bc=args.use_bc,
					 min_z_weight=args.min_z_weight, actor_lr=args.actor_lr,
					 risk_type=args.risk_type, risk_param=args.risk_param,
					 dist_penalty_type=args.dist_penalty_type,
					 lagrange_thresh=args.lag,
					 use_automatic_entropy_tuning=args.entropy_tuning,
					 device=device, wcql_cvar=args.wcql_cvar)
		# load model
		print("Load Model...")
		agent.load_model(os.path.join("MQNCQR-CVAR", args.env, "seed-2", "models", "epoch-1000"))
		print("done")

	# initial ensemble model
	state_size = np.prod(env.observation_space.shape)
	action_size = np.prod(env.action_space.shape)

	# initialize dynamics model
	env_model = ProbEnsemble(state_size, action_size, reward_size=1)
	env_model.to(device)

	# Imaginary Environment
	predict_env = PredictEnv(env_model, args.env)

	# Sampler Environment
	env_sampler = EnvSampler(env, max_path_length=args.epoch_length)

	# Initial replay buffer for env
	# if dataset is not None:
	# 	n = dataset['observations'].shape[0]
	# 	print(f"dataset name: {args.dataset}")
	# 	print(f"{args.env} dataset size {n}")
	# 	env_pool = ReplayMemory(n)
	# 	for i in range(n):
	# 		state, action, reward, next_state, done = dataset['observations'][i], dataset['actions'][i], \
	# 												  dataset['rewards'][i], dataset['next_observations'][i], dataset['terminals'][i]
	# 		env_pool.push(state, action, reward, next_state, done)
	# else:
	# 	env_pool = ReplayMemory(args.init_exploration_steps)
	# 	exploration_before_start(args, env_sampler, env_pool, agent,
	# 							 init_exploration_steps=args.init_exploration_steps)

	# Initial pool for model
	# rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
	# model_steps_per_epoch = int(args.rollout_length * rollouts_per_epoch)
	# new_pool_size = args.model_retain_epochs * model_steps_per_epoch
	# model_pool = ReplayMemory(new_pool_size)

	# if args.wandb:
	# 	wandb.init(project='wcql', group=args.env, name=run_name, config=args)
	#
	# with open(f"{args.save_dir}/args.json", "w") as f:
	# 	json.dump(vars(args), f, ensure_ascii=False, indent=4)

	# Train
	eval_policy(args, env_sampler, predict_env, agent)


if __name__ == '__main__':
	main()
