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


Base.@kwdef struct Config
  run_name::String = format(now(), "yy-mm-dd|HH:MM:SS")

  total_timesteps::Int = 500_000
  batch_size::Int = 128
  minibatch_size::Int32 = 32
  epochs::Int32 = 5

  # TODO: float32!
  lr::Float64 = 3e-4
  gamma::Float64 = 0.99
  lambda::Float64 = 0.95


  critic_loss_weight::Float64 = 0.5
  entropy_loss_weight::Float64 = 0.01
  clipping_epsilon::Float64 = 0.2
end


function gae(values::Vector{T}, rewards::Vector{T}, terminals::Vector{Bool}, γ::T, λ::T) where {T<:AbstractFloat}
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

  value_target = advantages + values[1:end-1]
  advantages, value_target
end

# todo instead of rb pass in all the datas
function learn_minibatch!(rb::Buffer.ReplayBuffer, advantages::Vector{<:AbstractFloat}, params::Flux.params, actor::Chain, critic::Chain, opt::Adam, config::Config)

  loss, gs = Flux.withgradient(params) do
    values = critic(data.state') |> vec
    advantage, value_target = ignore_derivatives() do
      gae(values, data.reward[2:end], data.terminal[2:end], config.gamma, config.lambda)
    end

    # clipping actor loss
    # TODO: make data 
    ac_dists = data.state' |> actor |> eachcol .|> d -> Categorical(d, check_args=false)
    new_log_probs = loglikelihood.(ac_dists, data.action) |> vec
    ratios = exp.(new_log_probs[1:end-1] .- data.log_prob[1:end-1])
    clipped_ratios = clamp.(ratios, 1 - config.clipping_epsilon, 1 + config.clipping_epsilon)
    # TODO: do we need to multiply both adv, or is it equivalent to take the min of the ratios
    #  and multiply by advantage after
    clipped_objective = min.(ratios .* advantage, clipped_ratios .* advantage)

    actor_loss = -mean(clipped_objective)

    # TODO: what values to t+1 or t?
    critic_loss = 0.5 * mean((values[2:end] .- value_target) .^ 2)

    ac_dist_entropy = mean(entropy.(ac_dists))

    actor_loss + config.critic_loss_weight * critic_loss - config.entropy_loss_weight * ac_dist_entropy
  end
  Flux.Optimise.update!(opt, params, gs)

  Buffer.clear(rb)

  loss
end
function learn!(rb::Buffer.ReplayBuffer, actor::Chain, critic::Chain, opt::Adam, config::Config)
  # data = Buffer.sample_and_remove(rb, rb.size; ordered=true)  # get all the data from the buffer
  data = rb.data
  # TODO:
  # [ ] minibatching
  # [ ] multiple train loops
  # [x] clipping loss
  # [x] value loss
  # [x] entropy

  values = critic(data.state') |> vec
  advantages, value_target = gae(values, data.reward[2:end], data.terminal[2:end], config.gamma, config.lambda)
  # TODO: calculate advantage out here so that you can completely mix the batch
  params = Flux.params(actor, critic)
  loss = 0
  for upd in 1:config.epochs
    # shuffle data
    inds = shuffle(1:rb.size-config.batch_size)
    for i in 1:config.minibatch_size:config.batch_size
      mb_inds = inds[i:i+config.minibatch_size]  # get minibatch num items
      loss += learn_minibatch!()
    end
  end
end

function ppo(config::Config)
  Logger.make_logger("ppo|$(config.run_name)")

  env = CartPoleEnv()  # TODO make env configurable through argparse

  actor, critic = Networks.make_actor_critic_nn(env)
  opt = Adam()  # one opt per network?

  transition = (
    state=rand(state_space(env)),
    action=rand(action_space(env)),
    log_prob=1.0,
    reward=1.0,
    terminal=true
  )
  rb = Buffer.ReplayBuffer(transition, config.batch_size * 2)
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

    transition = (
      state=obs,
      action=action,
      log_prob=loglikelihood(ac_dist, action),
      reward=reward(env),
      terminal=is_terminated(env)
    )
    Buffer.add!(rb, transition)

    # Recording episode statistics
    episode_return += transition.reward
    episode_length += 1

    if transition.terminal
      if rb.size > config.batch_size
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

const config = ConfigParser.argparse_struct(Config())
@time ppo(config)
