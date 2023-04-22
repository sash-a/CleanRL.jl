using ReinforcementLearningEnvironments: CartPoleEnv
using ReinforcementLearningBase: reset!, reward, state, is_terminated, action_space, state_space, AbstractEnv

using Flux
using StatsBase: sample, Weights, loglikelihood, mean, entropy
using Random: shuffle
using Distributions: Categorical
using ChainRulesCore

using Dates: now, format

include("../utils/replay_buffer.jl")
include("../utils/config_parser.jl")
include("../utils/logger.jl")
include("../utils/networks.jl")


function ppo()
  Logger.make_logger("ppo-crl-test")

  env = CartPoleEnv()  # TODO make env configurable through argparse

  actor = Chain(
    Dense(4, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 2),
    softmax,
  )
  critic = Chain(
    Dense(4, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 1)
  )

  function get_action_and_value(state, action=Nothing)
    logits = actor(state)
    probs = Categorical.(eachcol(logits), check_args=false)
    if action == Nothing
      action = rand.(probs)
    end
    log_prob = loglikelihood.(probs, action)

    action, log_prob, entropy.(logits), critic(state)
  end

  opt = Adam(2.5e-4, (0.9, 0.999), 1e-5)  # one opt per network?

  num_steps = 128
  total_timesteps = 500000
  batch_size = num_steps
  minibatch_size = 32
  num_updates = total_timesteps ÷ batch_size
  gamma = 0.99
  gae_lambda = 0.95
  update_epochs = 4
  clip_coef = 0.2
  ent_coeff = 0.01
  v_coef = 0.5

  # ALGO Logic: Storage setup
  obs = zeros((num_steps, 4))
  actions = zeros(Int64, num_steps, 1)
  logprobs = zeros(num_steps)
  rewards = zeros(num_steps)
  dones = zeros(num_steps)
  values = zeros(num_steps)

  global_step = 0
  episode_return = 0
  episode_length = 0

  start_time = time()
  reset!(env)

  next_obs = state(env)
  next_done = is_terminated(env)

  for update in 1:num_updates
    for step in 1:num_steps
      global_step += 1
      obs[step, :] = next_obs
      dones[step] = next_done

      action, log_prob, entropy, value = get_action_and_value(next_obs)
      values[step] = value[1]
      actions[step, :] = action
      logprobs[step] = log_prob[1]

      # todo cartpole expects an int, but other envs may expect a vec
      env(action[1])
      next_obs = deepcopy(state(env))
      next_done = deepcopy(is_terminated(env))
      rewards[step] = reward(env)
      episode_return += rewards[step]

      if next_done
        reset!(env)
        next_obs = state(env)
        next_done = is_terminated(env)
        @info "Episode Return" episode_return
        episode_return = 0
        episode_length = 0
      end
    end


    # bootstrap value if not done
    next_value = critic(next_obs)[1]
    returns = similar(rewards)
    advantages = similar(rewards)
    lastgaelam = 0
    for t in reverse(1:num_steps-1)
      if t == num_steps - 1
        nextnonterminal = 1.0 - next_done
        nextvalues = next_value
      else
        nextnonterminal = 1.0 - dones[t+1]
        nextvalues = values[t+1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
      end

      b_inds = 1:batch_size
      # do I need to copy buffer here?
      for epoch in 1:update_epochs
        b_inds = shuffle(b_inds)

        for start in 1:minibatch_size:batch_size
          end_ind = start + minibatch_size - 1
          mb_inds = b_inds[start:end_ind]

          _, newlogprob, entropy, newvalue = get_action_and_value(obs[mb_inds, :]', actions[mb_inds, :])
          logratio = dropdims(newlogprob; dims=2) - logprobs[mb_inds]
          ratio = exp.(logratio)

          # norm adv?
          # policy loss
          mb_advantages = advantages[mb_inds]
          pg_loss1 = -mb_advantages .* ratio
          pg_loss2 = -mb_advantages .* clamp.(ratio, 1 - clip_coef, 1 + clip_coef)
          pg_loss = mean(max.(pg_loss1, pg_loss2))

          # value loss
          # optional clip
          v_loss = 0.5 * mean((dropdims(newvalue; dims=1) - returns[mb_inds]) .^ 2)
          loss = pg_loss - ent_coeff * mean(entropy) + v_coef * v_loss
        end
      end
    end
  end
end

ppo()
