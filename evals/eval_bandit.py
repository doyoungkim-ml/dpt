import matplotlib.pyplot as plt

import numpy as np
import scipy
import torch
from IPython import embed


from ctrls.ctrl_bandit import (
    BanditTransformerController,
    GreedyOptPolicy,
    EmpMeanPolicy,
    OptPolicy,
    PessMeanPolicy,
    ThompsonSamplingPolicy,
    UCBPolicy,
)
from envs.bandit_env import BanditEnv, BanditEnvVec
from utils import convert_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def deploy_online(env, controller, horizon):
    context_states = torch.zeros((1, horizon, env.dx)).float().to(device)
    context_actions = torch.zeros((1, horizon, env.du)).float().to(device)
    context_next_states = torch.zeros((1, horizon, env.dx)).float().to(device)
    context_rewards = torch.zeros((1, horizon, 1)).float().to(device)

    cum_means = []
    for h in range(horizon):
        batch = {
            'context_states': context_states[:, :h, :],
            'context_actions': context_actions[:, :h, :],
            'context_next_states': context_next_states[:, :h, :],
            'context_rewards': context_rewards[:, :h, :],
        }

        controller.set_batch(batch)
        states_lnr, actions_lnr, next_states_lnr, rewards_lnr = env.deploy(
            controller)

        context_states[0, h, :] = convert_to_tensor(states_lnr[0])
        context_actions[0, h, :] = convert_to_tensor(actions_lnr[0])
        context_next_states[0, h, :] = convert_to_tensor(next_states_lnr[0])
        context_rewards[0, h, :] = convert_to_tensor(rewards_lnr[0])

        actions = actions_lnr.flatten()
        mean = env.get_arm_value(actions)

        cum_means.append(mean)

    return np.array(cum_means)


def deploy_online_vec(vec_env, controller, horizon, include_meta=False):
    num_envs = vec_env.num_envs
    # context_states = torch.zeros((num_envs, horizon, vec_env.dx)).float().to(device)
    # context_actions = torch.zeros((num_envs, horizon, vec_env.du)).float().to(device)
    # context_next_states = torch.zeros((num_envs, horizon, vec_env.dx)).float().to(device)
    # context_rewards = torch.zeros((num_envs, horizon, 1)).float().to(device)

    context_states = np.zeros((num_envs, horizon, vec_env.dx))
    context_actions = np.zeros((num_envs, horizon, vec_env.du))
    context_next_states = np.zeros((num_envs, horizon, vec_env.dx))
    context_rewards = np.zeros((num_envs, horizon, 1))

    cum_means = []
    print("Deplying online vectorized...")
    for h in range(horizon):
        batch = {
            'context_states': context_states[:, :h, :],
            'context_actions': context_actions[:, :h, :],
            'context_next_states': context_next_states[:, :h, :],
            'context_rewards': context_rewards[:, :h, :],
        }

        controller.set_batch_numpy_vec(batch)

        states_lnr, actions_lnr, next_states_lnr, rewards_lnr = vec_env.deploy(
            controller)

        context_states[:, h, :] = states_lnr
        context_actions[:, h, :] = actions_lnr
        context_next_states[:, h, :] = next_states_lnr
        context_rewards[:, h, :] = rewards_lnr[:,None]

        mean = vec_env.get_arm_value(actions_lnr)
        cum_means.append(mean)

    print("Deplyed online vectorized")
    
    cum_means = np.array(cum_means)
    if not include_meta:
        return cum_means
    else:
        meta = {
            'context_states': context_states,
            'context_actions': context_actions,
            'context_next_states': context_next_states,
            'context_rewards': context_rewards,
        }
        return cum_means, meta



def online(eval_trajs, model, n_eval, horizon, var, bandit_type):

    all_means = {}

    envs = []
    for i_eval in range(n_eval):
        print(f"Eval traj: {i_eval}")
        traj = eval_trajs[i_eval]
        means = traj['means']

        # TODO: Does bandit type need to be passed in?
        env = BanditEnv(means, horizon, var=var)
        envs.append(env)

    vec_env = BanditEnvVec(envs)
    
    controller = OptPolicy(
        envs,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T    
    assert cum_means.shape[0] == n_eval
    all_means['opt'] = cum_means


    controller = BanditTransformerController(
        model,
        sample=True,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    assert cum_means.shape[0] == n_eval
    all_means['Lnr'] = cum_means


    controller = EmpMeanPolicy(
        envs[0],
        online=True,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    assert cum_means.shape[0] == n_eval
    all_means['Emp'] = cum_means

    controller = UCBPolicy(
        envs[0],
        const=1.0,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    assert cum_means.shape[0] == n_eval
    all_means['UCB1.0'] = cum_means

    controller = ThompsonSamplingPolicy(
        envs[0],
        std=var,
        sample=True,
        prior_mean=0.5,
        prior_var=1/12.0,
        warm_start=False,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    assert cum_means.shape[0] == n_eval
    all_means['Thomp'] = cum_means


    all_means = {k: np.array(v) for k, v in all_means.items()}
    all_means_diff = {k: all_means['opt'] - v for k, v in all_means.items()}

    means = {k: np.mean(v, axis=0) for k, v in all_means_diff.items()}
    sems = {k: scipy.stats.sem(v, axis=0) for k, v in all_means_diff.items()}


    cumulative_regret = {k: np.cumsum(v, axis=1) for k, v in all_means_diff.items()}
    regret_means = {k: np.mean(v, axis=0) for k, v in cumulative_regret.items()}
    regret_sems = {k: scipy.stats.sem(v, axis=0) for k, v in cumulative_regret.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))


    for key in means.keys():
        if key == 'opt':
            ax1.plot(means[key], label=key, linestyle='--',
                    color='black', linewidth=2)
            ax1.fill_between(np.arange(horizon), means[key] - sems[key], means[key] + sems[key], alpha=0.2, color='black')
        else:
            ax1.plot(means[key], label=key)
            ax1.fill_between(np.arange(horizon), means[key] - sems[key], means[key] + sems[key], alpha=0.2)


    ax1.set_yscale('log')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Suboptimality')
    ax1.set_title('Online Evaluation')
    ax1.legend()


    for key in regret_means.keys():
        if key != 'opt':
            ax2.plot(regret_means[key], label=key)
            ax2.fill_between(np.arange(horizon), regret_means[key] - regret_sems[key], regret_means[key] + regret_sems[key], alpha=0.2)

    # ax2.set_yscale('log')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Regret Over Time')
    ax2.legend()




def offline(eval_trajs, model, n_eval, horizon, var, bandit_type):
    all_rs_lnr = []
    all_rs_greedy = []
    all_rs_opt = []
    all_rs_emp = []
    all_rs_pess = []
    all_rs_thmp = []

    num_envs = len(eval_trajs)

    tmp_env = BanditEnv(eval_trajs[0]['means'], horizon, var=var)
    context_states = np.zeros((num_envs, horizon, tmp_env.dx))
    context_actions = np.zeros((num_envs, horizon, tmp_env.du))
    context_next_states = np.zeros((num_envs, horizon, tmp_env.dx))
    context_rewards = np.zeros((num_envs, horizon, 1))


    envs = []

    print(f"Evaling offline horizon: {horizon}")

    for i_eval in range(n_eval):
        # print(f"Eval traj: {i_eval}")
        traj = eval_trajs[i_eval]
        means = traj['means']

        # TODO: Does bandit type need to be passed in?
        env = BanditEnv(means, horizon, var=var)
        envs.append(env)

        context_states[i_eval, :, :] = traj['context_states'][:horizon]
        context_actions[i_eval, :, :] = traj['context_actions'][:horizon]
        context_next_states[i_eval, :, :] = traj['context_next_states'][:horizon]
        context_rewards[i_eval, :, :] = traj['context_rewards'][:horizon,None]


    vec_env = BanditEnvVec(envs)
    batch = {
        'context_states': context_states,
        'context_actions': context_actions,
        'context_next_states': context_next_states,
        'context_rewards': context_rewards,
    }

    opt_policy = OptPolicy(envs, batch_size=num_envs)
    emp_policy = EmpMeanPolicy(envs[0], online=False, batch_size=num_envs)
    lnr_policy = BanditTransformerController(model, sample=False, batch_size=num_envs)
    thomp_policy = ThompsonSamplingPolicy(
        envs[0],
        std=var,
        sample=False,
        prior_mean=0.5,
        prior_var=1/12.0,
        warm_start=False,
        batch_size=num_envs)
    lcb_policy = PessMeanPolicy(
        envs[0],
        const=.8,
        batch_size=len(envs))


    opt_policy.set_batch_numpy_vec(batch)
    emp_policy.set_batch_numpy_vec(batch)
    thomp_policy.set_batch_numpy_vec(batch)
    lcb_policy.set_batch_numpy_vec(batch)
    lnr_policy.set_batch_numpy_vec(batch)
    
    _, _, _, rs_opt = vec_env.deploy_eval(opt_policy)
    _, _, _, rs_emp = vec_env.deploy_eval(emp_policy)
    _, _, _, rs_lnr = vec_env.deploy_eval(lnr_policy)
    _, _, _, rs_lcb = vec_env.deploy_eval(lcb_policy)
    _, _, _, rs_thmp = vec_env.deploy_eval(thomp_policy)


    baselines = {
        'opt': np.array(rs_opt),
        'lnr': np.array(rs_lnr),
        'emp': np.array(rs_emp),
        'thmp': np.array(rs_thmp),
        'lcb': np.array(rs_lcb),
    }    
    baselines_means = {k: np.mean(v) for k, v in baselines.items()}
    colors = plt.cm.viridis(np.linspace(0, 1, len(baselines_means)))
    plt.bar(baselines_means.keys(), baselines_means.values(), color=colors)
    plt.title(f'Mean Reward on {n_eval} Trajectories')


    return baselines


def offline_graph(eval_trajs, model, n_eval, horizon, var, bandit_type):
    horizons = np.linspace(1, horizon, 50, dtype=int)

    all_means = []
    all_sems = []
    for h in horizons:
        config = {
            'horizon': h,
            'var': var,
            'n_eval': n_eval,
            'bandit_type': bandit_type,
        }
        config['horizon'] = h
        baselines = offline(eval_trajs, model, **config)
        plt.clf()

        means = {k: np.mean(v, axis=0) for k, v in baselines.items()}
        sems = {k: scipy.stats.sem(v, axis=0) for k, v in baselines.items()}
        all_means.append(means)


    for key in means.keys():
        if not key == 'opt':
            regrets = [all_means[i]['opt'] - all_means[i][key] for i in range(len(horizons))]            
            plt.plot(horizons, regrets, label=key)
            plt.fill_between(horizons, regrets - sems[key], regrets + sems[key], alpha=0.2)

    plt.legend()
    plt.yscale('log')
    plt.xlabel('Dataset size')
    plt.ylabel('Suboptimality')
    config['horizon'] = horizon


def policy_entropy(eval_trajs, model, n_eval, horizon, var, bandit_type):
    """Compute entropy and cross-entropy of the learned policy."""
    num_envs = len(eval_trajs)

    tmp_env = BanditEnv(eval_trajs[0]['means'], horizon, var=var)
    context_states = np.zeros((num_envs, horizon, tmp_env.dx))
    context_actions = np.zeros((num_envs, horizon, tmp_env.du))
    context_next_states = np.zeros((num_envs, horizon, tmp_env.dx))
    context_rewards = np.zeros((num_envs, horizon, 1))

    envs = []
    optimal_actions = []

    for i_eval in range(n_eval):
        traj = eval_trajs[i_eval]
        means = traj['means']
        env = BanditEnv(means, horizon, var=var)
        envs.append(env)
        
        # Get optimal action for this environment
        optimal_actions.append(np.argmax(means))

        context_states[i_eval, :, :] = traj['context_states'][:horizon]
        context_actions[i_eval, :, :] = traj['context_actions'][:horizon]
        context_next_states[i_eval, :, :] = traj['context_next_states'][:horizon]
        context_rewards[i_eval, :, :] = traj['context_rewards'][:horizon,None]

    optimal_actions = np.array(optimal_actions)

    batch = {
        'context_states': torch.tensor(context_states).float().to(device),
        'context_actions': torch.tensor(context_actions).float().to(device),
        'context_next_states': torch.tensor(context_next_states).float().to(device),
        'context_rewards': torch.tensor(context_rewards).float().to(device),
    }

    # Get policy logits from the model
    lnr_policy = BanditTransformerController(model, sample=False, batch_size=num_envs)
    
    zeros = torch.zeros(num_envs, tmp_env.dx**2 + tmp_env.du + 1).float().to(device)
    batch['zeros'] = zeros
    
    # Get states for query (dummy states)
    states = torch.zeros((num_envs, tmp_env.dx)).float().to(device)
    batch['query_states'] = states
    
    logits = model(batch).cpu().detach().numpy()
    
    # Convert logits to probabilities
    probs = scipy.special.softmax(logits, axis=-1)
    
    # Compute entropy: H(p) = -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    entropies = -np.sum(probs * np.log(probs + eps), axis=-1)
    
    # Compute cross-entropy with optimal policy
    # Since optimal policy is deterministic, CE = -log(p[optimal_action])
    cross_entropies = -np.log(probs[np.arange(n_eval), optimal_actions] + eps)
    
    results = {
        'entropy': entropies,
        'cross_entropy': cross_entropies,
        'entropy_mean': np.mean(entropies),
        'entropy_sem': scipy.stats.sem(entropies),
        'cross_entropy_mean': np.mean(cross_entropies),
        'cross_entropy_sem': scipy.stats.sem(cross_entropies),
    }
    
    return results


def policy_entropy_online(eval_trajs, model, n_eval, horizon, var, bandit_type, include_baselines=False):
    """Compute entropy and cross-entropy at each timestep during online deployment."""
    envs = []
    optimal_actions = []
    
    for i_eval in range(n_eval):
        traj = eval_trajs[i_eval]
        means = traj['means']
        env = BanditEnv(means, horizon, var=var)
        envs.append(env)
        optimal_actions.append(np.argmax(means))
    
    optimal_actions = np.array(optimal_actions)
    tmp_env = envs[0]
    
    results = {}
    
    # Compute for learner (lnr)
    all_entropies = []
    all_cross_entropies = []
    
    context_states = torch.zeros((n_eval, horizon, tmp_env.dx)).float().to(device)
    context_actions = torch.zeros((n_eval, horizon, tmp_env.du)).float().to(device)
    context_next_states = torch.zeros((n_eval, horizon, tmp_env.dx)).float().to(device)
    context_rewards = torch.zeros((n_eval, horizon, 1)).float().to(device)
    
    for h in range(horizon):
        batch = {
            'context_states': context_states[:, :h, :],
            'context_actions': context_actions[:, :h, :],
            'context_next_states': context_next_states[:, :h, :],
            'context_rewards': context_rewards[:, :h, :],
        }
        
        zeros = torch.zeros(n_eval, tmp_env.dx**2 + tmp_env.du + 1).float().to(device)
        batch['zeros'] = zeros
        states = torch.zeros((n_eval, tmp_env.dx)).float().to(device)
        batch['query_states'] = states
        
        logits = model(batch).cpu().detach().numpy()
        probs = scipy.special.softmax(logits, axis=-1)
        
        eps = 1e-10
        entropies_t = -np.sum(probs * np.log(probs + eps), axis=-1)
        cross_entropies_t = -np.log(probs[np.arange(n_eval), optimal_actions] + eps)
        
        all_entropies.append(entropies_t)
        all_cross_entropies.append(cross_entropies_t)
        
        # Take actions
        for i, env in enumerate(envs):
            action_probs = probs[i]
            action_idx = np.random.choice(np.arange(tmp_env.du), p=action_probs)
            action = np.zeros(tmp_env.du)
            action[action_idx] = 1.0
            
            state = np.array([1])
            next_state, reward = env.transit(state, action)
            
            context_states[i, h, :] = convert_to_tensor(state)
            context_actions[i, h, :] = convert_to_tensor(action)
            context_next_states[i, h, :] = convert_to_tensor(next_state)
            context_rewards[i, h, :] = convert_to_tensor(reward)
    
    all_entropies = np.array(all_entropies).T
    all_cross_entropies = np.array(all_cross_entropies).T
    
    results['lnr'] = {
        'entropy_mean': np.mean(all_entropies, axis=0),
        'entropy_sem': scipy.stats.sem(all_entropies, axis=0),
        'cross_entropy_mean': np.mean(all_cross_entropies, axis=0),
        'cross_entropy_sem': scipy.stats.sem(all_cross_entropies, axis=0),
    }
    
    # Compute for baselines if requested
    if include_baselines:
        # Thompson Sampling
        results['thmp'] = _compute_baseline_entropy(envs, optimal_actions, horizon, var, 'thmp')
        # UCB
        results['ucb'] = _compute_baseline_entropy(envs, optimal_actions, horizon, var, 'ucb')
        # Empirical Mean
        results['emp'] = _compute_baseline_entropy(envs, optimal_actions, horizon, var, 'emp')
    
    return results


def _compute_baseline_entropy(envs, optimal_actions, horizon, var, baseline_type):
    """Helper to compute entropy for baseline policies."""
    n_eval = len(envs)
    tmp_env = envs[0]
    
    all_entropies = []
    all_cross_entropies = []
    
    # Create contexts for this baseline
    context_actions_np = np.zeros((n_eval, horizon, tmp_env.du))
    context_rewards_np = np.zeros((n_eval, horizon))
    
    # Reset environments
    for env in envs:
        env.reset()
    
    for h in range(horizon):
        # Get action probabilities from baseline
        if baseline_type == 'thmp':
            # Thompson Sampling - sample from posterior
            controller = ThompsonSamplingPolicy(tmp_env, std=var, sample=False,
                                              prior_mean=0.5, prior_var=1/12.0,
                                              warm_start=False, batch_size=n_eval)
            # Update posterior based on history
            probs = np.zeros((n_eval, tmp_env.du))
            for i in range(n_eval):
                # Compute posterior for this environment
                counts = np.zeros(tmp_env.du)
                rewards_sum = np.zeros(tmp_env.du)
                for t in range(h):
                    a_idx = np.argmax(context_actions_np[i, t])
                    counts[a_idx] += 1
                    rewards_sum[a_idx] += context_rewards_np[i, t]
                
                # Sample from posterior many times to get probability distribution
                prior_var = 1/12.0
                posterior_var = 1.0 / (1.0/prior_var + counts / (var**2 + 1e-10))
                posterior_mean = posterior_var * (0.5/prior_var + rewards_sum / (var**2 + 1e-10))
                
                # Monte Carlo to estimate policy distribution
                n_samples = 1000
                samples = np.random.normal(posterior_mean, np.sqrt(posterior_var), size=(n_samples, tmp_env.du))
                action_counts = np.bincount(np.argmax(samples, axis=1), minlength=tmp_env.du)
                probs[i] = action_counts / n_samples
                
        elif baseline_type == 'ucb':
            # UCB policy
            probs = np.zeros((n_eval, tmp_env.du))
            const = 1.0
            for i in range(n_eval):
                counts = np.zeros(tmp_env.du)
                rewards_sum = np.zeros(tmp_env.du)
                for t in range(h):
                    a_idx = np.argmax(context_actions_np[i, t])
                    counts[a_idx] += 1
                    rewards_sum[a_idx] += context_rewards_np[i, t]
                
                # UCB is deterministic - compute UCB values
                if h == 0:
                    # Choose uniformly at random
                    probs[i] = np.ones(tmp_env.du) / tmp_env.du
                else:
                    mean_rewards = rewards_sum / np.maximum(counts, 1)
                    ucb_values = mean_rewards + const * np.sqrt(np.log(h + 1) / np.maximum(counts, 1))
                    # Make nearly deterministic with small epsilon for numerical stability
                    probs[i] = 1e-6
                    probs[i, np.argmax(ucb_values)] = 1.0 - (tmp_env.du - 1) * 1e-6
                    
        elif baseline_type == 'emp':
            # Empirical Mean policy
            probs = np.zeros((n_eval, tmp_env.du))
            for i in range(n_eval):
                counts = np.zeros(tmp_env.du)
                rewards_sum = np.zeros(tmp_env.du)
                for t in range(h):
                    a_idx = np.argmax(context_actions_np[i, t])
                    counts[a_idx] += 1
                    rewards_sum[a_idx] += context_rewards_np[i, t]
                
                # Greedy w.r.t. empirical mean
                if h == 0 or np.all(counts == 0):
                    # Choose uniformly if no data
                    probs[i] = np.ones(tmp_env.du) / tmp_env.du
                else:
                    mean_rewards = rewards_sum / np.maximum(counts, 1)
                    # Nearly deterministic
                    probs[i] = 1e-6
                    best_arm = np.argmax(mean_rewards)
                    # If some arms haven't been tried, try them first
                    untried = np.where(counts == 0)[0]
                    if len(untried) > 0:
                        best_arm = untried[0]
                    probs[i, best_arm] = 1.0 - (tmp_env.du - 1) * 1e-6
        
        # Compute entropy and cross-entropy
        eps = 1e-10
        entropies_t = -np.sum(probs * np.log(probs + eps), axis=-1)
        cross_entropies_t = -np.log(probs[np.arange(n_eval), optimal_actions] + eps)
        
        all_entropies.append(entropies_t)
        all_cross_entropies.append(cross_entropies_t)
        
        # Sample actions and update context
        for i in range(n_eval):
            action_idx = np.random.choice(np.arange(tmp_env.du), p=probs[i])
            action = np.zeros(tmp_env.du)
            action[action_idx] = 1.0
            
            state = np.array([1])
            next_state, reward = envs[i].transit(state, action)
            
            context_actions_np[i, h] = action
            context_rewards_np[i, h] = reward
    
    all_entropies = np.array(all_entropies).T
    all_cross_entropies = np.array(all_cross_entropies).T
    
    return {
        'entropy_mean': np.mean(all_entropies, axis=0),
        'entropy_sem': scipy.stats.sem(all_entropies, axis=0),
        'cross_entropy_mean': np.mean(all_cross_entropies, axis=0),
        'cross_entropy_sem': scipy.stats.sem(all_cross_entropies, axis=0),
    }
