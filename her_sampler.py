import numpy as np


def make_sample_her_transitions(distance_threshold):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    replay_k = 4
    future_p = 1 - (1. / (1 + replay_k))  # --> 0.8

    def _sample_her_transitions(episode_batch, batch_size):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]  # rollout length
        num_rollouts = episode_batch['u'].shape[0]  # num workers
        # num_rollouts * (rollout_horizon - 1) --> total steps per cycle
        # the minus 1 is so that you don't sample the very last transitions, since you can't do HER with it

        # SUMMARY of these lines: for each cycle, randomly pick a bunch of transitions, equal to 1 cycle size
        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, num_rollouts, batch_size)  # 0 through num_rollouts array. length=steps per cycle
        t_samples = np.random.randint(T, size=batch_size)  # array of random numbers in range(rollout_horizon), length=steps per cycle
        # the episode_idxs indexes into which worker you want -- pick only one
        # t_samples picks the timestep in the rollout
        # with both, you pick one particular timestep for one rollout. This only applies to one cycle at a time, and gets repeated for each cycle
        # transitions is sampled from your episode -- for each cycle, randomly pick a subset of steps and workers
        # end shape: (steps per cycle, num_cycles)  Note: here steps per cycle is (rollout_horizon - 1) * num_rollouts
        transitions = {key: episode_batch[key][episode_idxs, t_samples]
                       for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.

        # SUMMARY: pick a bunch of times in the future for each previously selected transition
        # indices in one cycle that are selected for HER replay.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        # pick some random time in the future between the time selected (t_samples) and the end of the rollout
        # shape of future_offset = same as transitions
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        # shape of future t = (steps per cycle * future_p, )
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        # replace goal with achieved goal according to previously selected indices.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Re-compute reward since we may have substituted the goal.
        distance = np.linalg.norm(transitions['ag_2'] - transitions['g'], axis=-1)
        transitions['r'] = -(distance > distance_threshold).astype(np.float32)

        assert(transitions['u'].shape[0] == batch_size)
        return transitions

    return _sample_her_transitions
