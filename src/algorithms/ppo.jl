using ReinforcementLearningEnvironments: CartPoleEnv
using ReinforcementLearningBase: reset!, reward, state, is_terminated, action_space, state_space, AbstractEnv

using Flux
using StatsBase: sample, Weights, loglikelihood, mean
using Distributions: Categorical

using Dates: now, format

include("../utils/buffers.jl")
include("../utils/config_parser.jl")
include("../utils/logger.jl")
include("../utils/networks.jl")


Base.@kwdef struct Config
  run_name::String = format(now(), "yy-mm-dd|HH:MM:SS")

  total_timesteps::Int = 500_000
  min_replay_size::Int = 512

  lr::Float32 = 3e-4
  gamma::Float32 = 0.99
  lambda::Float32 = 0.95
  epochs::Int32 = 5
  minibatch_szie::Int32 = 32

  critic_loss_weight::Float32 = 0.9
  entropy_loss_weight::Float32 = 0.01
  clipping_epsilon::Float32 = 0.2
end


function gae(values::Vector{T}, rewards::Vector{T}, terminals::Vector{T}, γ::T, λ::T) where {T<:AbstractFloat}
  """
  Generalized advantage estimation.

  Args:
    values: [0, k]
    rewards: [1, k]
    terminals: [1,k]
    γ: gamma/discount
    λ: gae lambda

  Returns: 
   advatages [0, k-1]
  """
  advantages = similar(rewards)
  gae = 0.0
  for t in length(rewards):-1:1
    δ = rewards[t] + terminals[t] * values[t+1] - values[t]
    gae = δ + γ * λ * terminals[t] * gae
    advantages[t] = gae
  end

  # value target = advantages + values[1:end-1]
  advantages
end

function learn!(rb::Buffers.ReplayQueue{Buffers.PGTransition}, actor::Chain, critic::Chain, opt::Adam, config::Config)
  data = sample(rb, length(rb.data))  # get all the data from the buffer
  # TODO:
  # [ ] minibatching
  # [ ] multiple train loops
  # [ ] clipping loss
  # [x] value loss
  # [ ] entropy
  params = Flux.params(actor, critic)
  loss, gs = Flux.withgradient(params) do
    # chain rules: ignore derivative for stopping grads
    # critic loss
    values = critic(data.states)
    advantage = gae(values, data.rewards[2:end], data.terminals[2:end], config.gamma, config.lambda)

    critic_loss = mean(advantage .^ 2)

    # actor loss -> TODO: ppo's clipping loss
    ac_dists = data.states |> actor |> eachcol .|> d -> Categorical(d, check_args=false)
    log_probs = loglikelihood.(ac_dists, data.actions)
    actor_loss = -mean(log_probs .* advantage)

    # TODO: entropy
    entropy = 0

    actor_loss - config.critic_loss_weight * critic_loss + config.entropy_loss_weight * entropy
  end
  Flux.Optimise.update!(opt, params, gs)

  loss
end

function ppo()
  config = ConfigParser.argparse_struct(Config())
  Logger.make_logger("ppo|$(config.run_name)")

  env = CartPoleEnv()  # TODO make env configurable through argparse

  actor, critic = Networks.make_actor_critic_nn(env)
  opt = Adam()  # one opt per network?

  rb = Buffers.ReplayQueue{Buffers.PGTransition}()

  episode_return = 0
  episode_length = 0

  start_time = time()
  reset!(env)
  for global_step in 1:config.total_timesteps
    # state needs to be coppied otherwise each loop will overwrite ref in replay buffer
    obs = deepcopy(state(env))
    ac_dist = Categorical(actor(obs))
    action = rand(ac_dist)

    env(action)

    transition = Buffers.PGTransition(
      obs,
      action,  # add ac_dist for ppo to compute old log prob
      reward(env),
      is_terminated(env)
    )
    Buffers.add!(rb, transition)

    # Recording episode statistics
    episode_return += reward(env)
    episode_length += 1

    if is_terminated(env)
      if length(rb.data) > config.min_replay_size
        # training
        loss = learn!(rb, actor, critic, opt, config)
        # logging
        steps_per_second = trunc(global_step / (time() - start_time))
        @info "Training Statistics" loss steps_per_second
        @info "Episode Statistics" episode_return episode_length
      end

      # reset counters
      episode_length, episode_return = 0, 0
      reset!(env)
    end

  end
end

# @time ppo()
