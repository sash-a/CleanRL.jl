using ReinforcementLearningEnvironments: CartPoleEnv
using ReinforcementLearningBase: reset!, reward, state, is_terminated, action_space, state_space, AbstractEnv
using Flux

using StatsBase: sample, Weights, loglikelihood, mean
using Distributions: Categorical

using Dates: now, format

include("../utils/buffers.jl")
include("../utils/config_parser.jl")
include("../utils/logger.jl")


Base.@kwdef struct Config
  run_name::String = format(now(), "yy-mm-dd|HH:MM:SS")

  log_frequencey::Int = 1000

  total_timesteps::Int = 1_000_000

  batch_size::Int64 = 120
  gamma::Float64 = 0.99
end

function make_nn(env::AbstractEnv)
  in_size = length(state_space(env).s)
  out_size = length(action_space(env).s)
  ob_net = Chain(Dense(in_size, 120, relu), Dense(120, 84, relu))
  actor = Chain(ob_net, Dense(84, out_size), softmax)
  critic = Chain(ob_net, Dense(84, 1))

  actor, critic
end

function discounted_future_rewards(rewards::Vector{T}, terminals::Vector{Bool}, γ::T) where {T<:AbstractFloat}
  # TODO: this probably doesn't work with sequences/incomplete episodes
  #  If an episode doesn't complete then early actions will not get the full reward 
  #  from an episode as it will not be in the sequence...
  #  Fix this with a final value + finding the last terminal??
  future_rewards = zeros(eltype(rewards), size(rewards))
  future_rewards[1] = last(rewards) * last(terminals)

  # would be nice if the reverse was also a view
  reversed_rewards = reverse(@view rewards[1:end-1])
  reversed_terminals = reverse(@view terminals[1:end-1])
  for (i, (r, t)) in enumerate(zip(reversed_rewards, reversed_terminals))
    future_rewards[i+1] += t ? 0 : r + γ * future_rewards[i]
  end

  reverse(future_rewards)
end

# TODO:
#  training freq
#  logging freq
#  harder envs
function a2c()
  config = ConfigParser.argparse_struct(Config())
  Logger.make_logger("a2c|$(config.run_name)")

  env = CartPoleEnv()  # TODO make env configurable through argparse

  actor, critic = make_nn(env)
  opt = ADAM()

  rb = Buffers.ReplayQueue{Buffers.PGTransition}()

  episode_return = 0
  episode_length = 0

  start_time = time()
  reset!(env)
  for global_step in 1:config.total_timesteps
    obs = deepcopy(state(env))  # state needs to be coppied otherwise state and next_state is the same

    ac_dist = Categorical(actor(obs))
    action = rand(ac_dist)

    env(action)

    transition = Buffers.PGTransition(
      obs,
      action,
      reward(env),
      is_terminated(env)
    )
    Buffers.add!(rb, transition)

    # Recording episode statistics
    episode_return += reward(env)
    episode_length += 1

    if is_terminated(env)
      @info "Episode Statistics" episode_return episode_length

      reset!(env)

      # TODO: method?
      #  learn at regular interval instead of after episode?

      # learn
      data = sample(rb, episode_length)
      values = critic(data.states)
      discounted_rewards = discounted_future_rewards(data.rewards, data.terminals, config.gamma)
      advantage = discounted_rewards - vec(values)

      # actor update
      actor_params = Flux.params(actor)
      actor_loss, actor_gs = Flux.withgradient(actor_params) do
        ac_dists = data.states |> actor |> eachcol .|> d -> Categorical(d, check_args=false)
        log_probs = loglikelihood.(ac_dists, data.actions)
        -mean(log_probs .* advantage)
      end
      Flux.Optimise.update!(opt, actor_params, actor_gs)

      # critic update
      critic_params = Flux.params(critic)
      critic_loss, critic_gs = Flux.withgradient(critic_params) do
        # TODO: how to get advantage out of this block so only need to calc it once?
        values = critic(data.states)
        advantage = discounted_rewards - vec(values)
        mean(advantage .^ 2)  # TODO: this loss is *really* high, can't be normal?
      end
      Flux.Optimise.update!(opt, critic_params, critic_gs)

      # reset these at end of episode
      episode_length, episode_return = 0, 0

      steps_per_second = trunc(global_step / (time() - start_time))
      @info "Training Statistics" actor_loss critic_loss steps_per_second
    end
  end
end

@time a2c()
