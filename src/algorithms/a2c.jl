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

  gamma::Float64 = 0.99
end


function discounted_future_rewards(rewards::Vector{T}, terminals::Vector{Bool}, final_value::T, γ::T) where {T<:AbstractFloat}
  future_rewards = zeros(eltype(rewards), size(rewards))
  future_rewards[1] = last(terminals) ? 0.0 : last(rewards) + γ * final_value

  reversed_rewards = reverse(@view rewards[1:end-1])  # would be nice if the reverse was also a view
  reversed_terminals = reverse(@view terminals[1:end-1])
  for (i, (r, t)) in enumerate(zip(reversed_rewards, reversed_terminals))
    future_rewards[i+1] += t ? 0.0 : r + γ * future_rewards[i]
  end

  reverse(future_rewards)
end

# TODO:
#  test harder envs
#  mulitple V train steps
#  normalise advantage
#  lr
function a2c()
  config = ConfigParser.argparse_struct(Config())
  Logger.make_logger("a2c|$(config.run_name)")

  env = CartPoleEnv()  # TODO make env configurable through argparse

  actor, critic = Networks.make_actor_critic_nn(env)
  opt = ADAM()  # two opts?

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
      action,
      reward(env),
      is_terminated(env)
    )
    Buffers.add!(rb, transition)

    # Recording episode statistics
    episode_return += reward(env)
    episode_length += 1

    if is_terminated(env)
      if length(rb.data) > config.min_replay_size  # training
        data = sample(rb, length(rb.data))  # get all the data from the buffer
        final_value = data.states |> eachcol |> last |> critic |> first
        discounted_rewards = discounted_future_rewards(data.rewards, data.terminals, final_value, config.gamma)

        advantage = []
        params = Flux.params(actor, critic)
        loss, gs = Flux.withgradient(params) do
          # critic loss
          values = critic(data.states)
          advantage = discounted_rewards - vec(values)
          critic_loss = mean(advantage .^ 2)

          # actor loss
          ac_dists = data.states |> actor |> eachcol .|> d -> Categorical(d, check_args=false)
          log_probs = loglikelihood.(ac_dists, data.actions)
          actor_loss = -mean(log_probs .* advantage)

          actor_loss + critic_loss
        end
        Flux.Optimise.update!(opt, params, gs)


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

@time a2c()
