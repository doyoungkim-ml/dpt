import matplotlib.pyplot as plt

import numpy as np
import scipy
import torch
from IPython import embed


from ctrls.ctrl_bandit import (
    BanditTransformerController,
    EmpMeanPolicy,
    OptPolicy,
    ThompsonSamplingPolicy,
    LinUCBPolicy,
)
from envs.bandit_env import BanditEnv, BanditEnvVec, LinearBanditEnv
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



def online(eval_trajs, model, n_eval, horizon, var):

    all_means = {}

    envs = []
    for i_eval in range(n_eval):
        print(f"Eval traj: {i_eval}")
        traj = eval_trajs[i_eval]
        means = traj['means']

        # TODO: Does bandit type need to be passed in?
        env = LinearBanditEnv(traj['theta'], traj['arms'], horizon, var=var)
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



    controller = ThompsonSamplingPolicy(
        envs[0],
        std=var,
        sample=True,
        prior_mean=0.0,
        prior_var=1.0,
        warm_start=False,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    assert cum_means.shape[0] == n_eval
    all_means['Thomp'] = cum_means

    controller = LinUCBPolicy(
        envs[0],
        const=1.0,
        batch_size=len(envs)
    )
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    assert cum_means.shape[0] == n_eval
    all_means['LinUCB'] = cum_means

    
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




def offline(eval_trajs, model, n_eval, horizon, var):
    all_rs_lnr = []
    all_rs_greedy = []
    all_rs_opt = []
    all_rs_emp = []
    all_rs_pess = []
    all_rs_thmp = []

    num_envs = len(eval_trajs)

    # tmp_env = LinearBanditEnv(eval_trajs[0]['means'], horizon, var=var)
    tmp_env = LinearBanditEnv(eval_trajs[0]['theta'], eval_trajs[0]['arms'], horizon, var=var)
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
        env = LinearBanditEnv(traj['theta'], traj['arms'], horizon, var=var)
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
    lnr_policy = BanditTransformerController(model, sample=False, batch_size=num_envs)
    thomp_policy = ThompsonSamplingPolicy(
        envs[0],
        std=var,
        sample=False,
        prior_mean=0,
        prior_var=1.0,
        warm_start=False,
        batch_size=num_envs)
    linreg_policy = LinUCBPolicy(
        envs[0],
        const=0.0,
        batch_size=num_envs
    )

    opt_policy.set_batch_numpy_vec(batch)
    thomp_policy.set_batch_numpy_vec(batch)
    lnr_policy.set_batch_numpy_vec(batch)
    linreg_policy.set_batch_numpy_vec(batch)
    
    _, _, _, rs_opt = vec_env.deploy_eval(opt_policy)
    _, _, _, rs_lnr = vec_env.deploy_eval(lnr_policy)
    _, _, _, rs_thmp = vec_env.deploy_eval(thomp_policy)
    _, _, _, rs_linreg = vec_env.deploy_eval(linreg_policy)


    baselines = {
        'opt': np.array(rs_opt),
        'lnr': np.array(rs_lnr),
        'thmp': np.array(rs_thmp),
        'linreg': np.array(rs_linreg),
    }    
    baselines_means = {k: np.mean(v) for k, v in baselines.items()}
    colors = plt.cm.viridis(np.linspace(0, 1, len(baselines_means)))
    plt.bar(baselines_means.keys(), baselines_means.values(), color=colors)
    plt.title(f'Mean Reward on {n_eval} Trajectories')


    return baselines


def offline_graph(eval_trajs, model, n_eval, horizon, var):
    horizons = np.linspace(1, horizon, horizon, dtype=int)

    all_means = []
    all_sems = []

    all_subopt_means = []
    all_subopt_sems = []
    for h in horizons:
        config = {
            'horizon': h,
            'var': var,
            'n_eval': n_eval,
        }
        config['horizon'] = h
        baselines = offline(eval_trajs, model, **config)
        plt.clf()

        subopt = {k: baselines['opt'] - v for k, v in baselines.items()}
        subopt_means = {k: np.mean(v) for k, v in subopt.items()}
        subopt_sems = {k: scipy.stats.sem(v) for k, v in subopt.items()}

        means = {k: np.mean(v, axis=0) for k, v in baselines.items()}
        sems = {k: scipy.stats.sem(v, axis=0) for k, v in baselines.items()}
        
        all_means.append(means)
        all_sems.append(sems)
        all_subopt_means.append(subopt_means)
        all_subopt_sems.append(subopt_sems)

    all_subopt_means = np.array(all_subopt_means)
    all_subopt_sems = np.array(all_subopt_sems)


    for key in means.keys():
        if not key == 'opt':
            subopt_mean = [all_subopt_means[i][key] for i in range(len(horizons))]
            subopt_sem = [all_subopt_sems[i][key] for i in range(len(horizons))]
            subopt_mean = np.array(subopt_mean)
            subopt_sem = np.array(subopt_sem)

            plt.plot(horizons, subopt_mean, label=key)
            plt.fill_between(horizons, subopt_mean - subopt_sem, subopt_mean + subopt_sem, alpha=0.2)


    plt.legend()
    plt.yscale('log')
    plt.xlabel('Dataset size')
    plt.ylabel('Suboptimality')
    config['horizon'] = horizon


def policy_entropy_online(eval_trajs, model, n_eval, horizon, var, include_baselines=False):
    """Compute entropy and cross-entropy at each timestep during online deployment."""
    envs = []
    optimal_actions = []
    
    for i_eval in range(n_eval):
        traj = eval_trajs[i_eval]
        env = LinearBanditEnv(traj['theta'], traj['arms'], horizon, var=var)
        envs.append(env)
        optimal_actions.append(np.argmax(traj['means']))
    
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
        # Thompson Sampling for linear bandits
        results['thmp'] = _compute_linear_baseline_entropy(envs, optimal_actions, horizon, var, 'thmp')
        # LinUCB (essentially least-squares with const=0)
        results['linucb'] = _compute_linear_baseline_entropy(envs, optimal_actions, horizon, var, 'linucb')
    
    return results


def _compute_linear_baseline_entropy(envs, optimal_actions, horizon, var, baseline_type):
    """Helper to compute entropy for linear bandit baseline policies."""
    n_eval = len(envs)
    tmp_env = envs[0]
    d = len(tmp_env.theta)  # Feature dimension
    K = tmp_env.du  # Number of arms
    arms = tmp_env.arms  # (K, d) array of arm features
    
    all_entropies = []
    all_cross_entropies = []
    
    # Initialize for each environment
    if baseline_type == 'thmp':
        # Thompson Sampling: Bayesian linear regression
        # Prior: theta ~ N(0, I)
        prior_cov = np.eye(d)
        prior_mean = np.zeros(d)
        
        # Posterior parameters for each environment
        cov_invs = [np.eye(d) for _ in range(n_eval)]  # Precision matrices
        cov_mean_prods = [np.zeros(d) for _ in range(n_eval)]  # Precision * mean
        
    elif baseline_type == 'linucb':
        # LinUCB with const=1.0
        const = 1.0
        # Design matrix A and vector b for each environment
        A_matrices = [np.eye(d) for _ in range(n_eval)]
        b_vectors = [np.zeros(d) for _ in range(n_eval)]
    
    # Reset environments
    for env in envs:
        env.reset()
    
    for h in range(horizon):
        probs = np.zeros((n_eval, K))
        
        if baseline_type == 'thmp':
            # Thompson Sampling: Sample theta from posterior and compute probabilities
            for i in range(n_eval):
                # Posterior covariance and mean
                try:
                    posterior_cov = np.linalg.inv(cov_invs[i])
                    posterior_mean = posterior_cov @ cov_mean_prods[i]
                except:
                    # Fallback to prior if singular
                    posterior_cov = prior_cov
                    posterior_mean = prior_mean
                
                # Monte Carlo sampling to estimate policy distribution
                n_samples = 1000
                sampled_thetas = np.random.multivariate_normal(posterior_mean, posterior_cov, size=n_samples)
                
                # For each sample, compute best arm
                rewards_samples = sampled_thetas @ arms.T  # (n_samples, K)
                best_arms = np.argmax(rewards_samples, axis=1)
                
                # Estimate probability distribution
                action_counts = np.bincount(best_arms, minlength=K)
                probs[i] = action_counts / n_samples
                
        elif baseline_type == 'linucb':
            # LinUCB: Deterministic based on UCB values
            for i in range(n_eval):
                try:
                    A_inv = np.linalg.inv(A_matrices[i])
                    theta_hat = A_inv @ b_vectors[i]
                except:
                    # Fallback if singular
                    theta_hat = np.zeros(d)
                    A_inv = np.eye(d)
                
                # Compute UCB for each arm
                ucb_values = np.zeros(K)
                for a in range(K):
                    arm_feature = arms[a]
                    mean_reward = arm_feature @ theta_hat
                    uncertainty = const * np.sqrt(arm_feature @ A_inv @ arm_feature)
                    ucb_values[a] = mean_reward + uncertainty
                
                # Nearly deterministic policy
                best_arm = np.argmax(ucb_values)
                probs[i] = 1e-6
                probs[i, best_arm] = 1.0 - (K - 1) * 1e-6
        
        # Compute entropy and cross-entropy
        eps = 1e-10
        entropies_t = -np.sum(probs * np.log(probs + eps), axis=-1)
        cross_entropies_t = -np.log(probs[np.arange(n_eval), optimal_actions] + eps)
        
        all_entropies.append(entropies_t)
        all_cross_entropies.append(cross_entropies_t)
        
        # Sample actions and observe rewards, then update
        for i in range(n_eval):
            action_idx = np.random.choice(np.arange(K), p=probs[i])
            action = np.zeros(K)
            action[action_idx] = 1.0
            
            state = np.array([1])
            next_state, reward = envs[i].transit(state, action)
            
            arm_feature = arms[action_idx]
            
            if baseline_type == 'thmp':
                # Update posterior
                noise_var = var**2 + 1e-10
                cov_invs[i] += np.outer(arm_feature, arm_feature) / noise_var
                cov_mean_prods[i] += arm_feature * reward / noise_var
                
            elif baseline_type == 'linucb':
                # Update A and b
                A_matrices[i] += np.outer(arm_feature, arm_feature)
                b_vectors[i] += arm_feature * reward
    
    all_entropies = np.array(all_entropies).T
    all_cross_entropies = np.array(all_cross_entropies).T
    
    return {
        'entropy_mean': np.mean(all_entropies, axis=0),
        'entropy_sem': scipy.stats.sem(all_entropies, axis=0),
        'cross_entropy_mean': np.mean(all_cross_entropies, axis=0),
        'cross_entropy_sem': scipy.stats.sem(all_cross_entropies, axis=0),
    }


def policy_entropy(eval_trajs, model, n_eval, horizon, var):
    """Compute entropy and cross-entropy of the learned policy."""
    num_envs = len(eval_trajs)

    tmp_env = LinearBanditEnv(eval_trajs[0]['theta'], eval_trajs[0]['arms'], horizon, var=var)
    context_states = np.zeros((num_envs, horizon, tmp_env.dx))
    context_actions = np.zeros((num_envs, horizon, tmp_env.du))
    context_next_states = np.zeros((num_envs, horizon, tmp_env.dx))
    context_rewards = np.zeros((num_envs, horizon, 1))

    envs = []
    optimal_actions = []

    for i_eval in range(n_eval):
        traj = eval_trajs[i_eval]
        env = LinearBanditEnv(traj['theta'], traj['arms'], horizon, var=var)
        envs.append(env)
        
        # Get optimal action for this environment
        optimal_actions.append(np.argmax(traj['means']))

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
