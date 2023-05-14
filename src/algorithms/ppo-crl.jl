using ReinforcementLearningEnvironments: CartPoleEnv
using Base: global_logger
using ReinforcementLearningBase: reset!, reward, state, is_terminated, action_space, state_space, AbstractEnv

using Flux
using StatsBase: sample, Weights, loglikelihood, mean, entropy, std
using Random: shuffle
using Distributions: Categorical
using ChainRulesCore

using Dates: now, format

include("../utils/replay_buffer.jl")
include("../utils/config_parser.jl")
include("../utils/logger.jl")
include("../utils/networks.jl")

function get_action_and_value(state::AbstractVecOrMat{Float64}, actor::Chain, critic::Chain, action::Union{VecOrMat{Int64},Nothing}=nothing)
  # @show any(isnan.(state))
  logits = actor(state)
  # probs = Categorical.(eachcol(logits), check_args=false)
  # @show typeof(logits) eltype(logits) logits
  # @show eachcol(logits) eachcol(logits).parent
  # probs = Categorical(logits)
  # probs = mapslices(Categorical, logits; dims=1)
  # @show any(isnan.(logits))
  probs = Categorical.(collect.(eachcol(logits)))  # the collect here is slow :(
  if action === nothing
    action = rand.(probs)
  end

  log_prob = loglikelihood.(probs, action)

  action, log_prob, entropy.(logits), critic(state)
end

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

  batch_size = num_steps = 512
  total_timesteps = 500_000
  num_minibatches = 4
  minibatch_size = batch_size ÷ num_minibatches
  num_updates = total_timesteps ÷ batch_size
  update_epochs = 4
  lr = 2.5e-4
  gamma = 0.99
  gae_lambda = 0.95
  clip_coef = 0.2
  ent_coeff = 0.01
  v_coef = 0.5

  opt = Flux.Optimiser(ClipNorm(0.5), Adam(2.5e-4, (0.9, 0.999), 1e-5))  # one opt per network?

  # ALGO Logic: Storage setup
  obs = zeros(Float64, (num_steps, 4))
  actions = zeros(Int64, num_steps)
  logprobs = zeros(Float64, num_steps)
  rewards = zeros(Float64, num_steps)
  dones = zeros(Bool, num_steps)
  values = zeros(Float64, num_steps)

  global_step = 0
  episode_return = 0
  episode_length = 0

  start_time = time()
  reset!(env)

  next_obs = state(env)
  next_done = is_terminated(env)

  for update in 1:num_updates
    frac = 1.0 - (update - 1.0) / num_updates
    opt.os[2].eta = frac * lr
    for step in 1:num_steps
      global_step += 1
      obs[step, :] = next_obs
      dones[step] = next_done

      action, log_prob, entropy, value = get_action_and_value(next_obs, actor, critic)
      values[step] = value[1]
      # actions[step, :] = action
      actions[step] = action[1]
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
        steps_per_sec = global_step / (time() - start_time)
        @info "Episode Statistics" episode_return global_step steps_per_sec
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
    end

    b_inds = 1:batch_size
    # do I need to copy buffer here?

    for epoch in 1:update_epochs
      b_inds = shuffle(b_inds)

      params = Flux.params(actor, critic)
      for start in 1:minibatch_size:batch_size
        loss, gs = Flux.withgradient(params) do
          end_ind = start + minibatch_size - 1
          mb_inds = b_inds[start:end_ind]

          _, newlogprob, entropy, newvalue = get_action_and_value(obs[mb_inds, :]', actor, critic, actions[mb_inds, :])
          logratio = dropdims(newlogprob; dims=2) - logprobs[mb_inds]
          ratio = exp.(logratio)

          # todo: norm adv?
          # policy loss
          mb_advantages = (advantages[mb_inds] .- mean(advantages[mb_inds])) / (std(advantages[mb_inds]) .+ 1e-8)
          pg_loss1 = -mb_advantages .* ratio
          pg_loss2 = -mb_advantages .* clamp.(ratio, 1 - clip_coef, 1 + clip_coef)
          pg_loss = mean(max.(pg_loss1, pg_loss2))

          # value loss
          # todo: optional clip
          # v_loss = 0.5 * mean((dropdims(newvalue; dims=1) - returns[mb_inds]) .^ 2)
          newvalue = dropdims(newvalue; dims=1)
          v_loss_unclipped = mean(newvalue - returns[mb_inds] .^ 2)
          v_clipped = values[mb_inds] + clamp.(newvalue - values[mb_inds], -clip_coef, clip_coef)
          v_loss_clipped = (v_clipped - returns[mb_inds]) .^ 2
          v_loss_max = max.(v_loss_unclipped, v_loss_clipped)
          v_loss = 0.5 * mean(v_loss_max)


          pg_loss - ent_coeff * mean(entropy) + v_coef * v_loss
        end

        @info "Training Statistics" loss
        Flux.Optimise.update!(opt, params, gs)
      end
    end
  end
end

# ppo()
