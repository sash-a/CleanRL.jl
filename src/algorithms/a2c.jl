using ReinforcementLearningEnvironments: CartPoleEnv
using ReinforcementLearningBase: reset!, reward, state, is_terminated, action_space, state_space, AbstractEnv

using Flux
using StatsBase: sample, Weights, loglikelihood, mean
using Distributions: Categorical

using Dates: now, format

include("../utils/buffers.jl")
include("../utils/buffers2.jl")
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
  opt = ADAM()

  transition = (
    state=rand(state_space(env)),
    action=rand(action_space(env)),
    reward=1.0,
    terminal=true
  )
  # TODO: transition sampler
  rb = Buffer2.ReplayBuffer(transition, config.min_replay_size * 2)

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

    transition = ( #Buffers.PGTransition(
      state=obs,
      action=action,
      reward=reward(env),
      terminal=is_terminated(env)
    )
    Buffer2.add!(rb, transition)

    # Recording episode statistics
    episode_return += reward(env)
    episode_length += 1

    if is_terminated(env)
      if rb.size > config.min_replay_size  # training
        data = Buffer2.sample_and_remove(rb, rb.size; ordered=true)  # get all the data from the buffer
        final_value = data.state' |> eachcol |> last |> critic |> first
        discounted_rewards = discounted_future_rewards(vec(data.reward), vec(data.terminal), final_value, config.gamma)

        # critic update
        advantage = []
        critic_params = Flux.params(critic)
        critic_loss, critic_gs = Flux.withgradient(critic_params) do
          values = critic(data.state')
          advantage = discounted_rewards - vec(values)
          mean(advantage .^ 2)  # TODO: this loss is *really* high, is that normal?
        end
        Flux.Optimise.update!(opt, critic_params, critic_gs)

        # actor update
        actor_params = Flux.params(actor)
        actor_loss, actor_gs = Flux.withgradient(actor_params) do
          ac_dists = data.state' |> actor |> eachcol .|> d -> Categorical(d, check_args=false)
          log_probs = loglikelihood.(ac_dists, data.action)
          -mean(log_probs .* advantage)
        end
        Flux.Optimise.update!(opt, actor_params, actor_gs)

        # logging
        steps_per_second = trunc(global_step / (time() - start_time))
        @info "Training Statistics" actor_loss critic_loss steps_per_second
        @info "Episode Statistics" episode_return episode_length
      end

      # reset counters
      episode_length, episode_return = 0, 0
      reset!(env)
    end

  end
end

@time a2c()
