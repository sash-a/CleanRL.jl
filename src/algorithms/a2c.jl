using ReinforcementLearningEnvironments: CartPoleEnv
using ReinforcementLearningBase: reset!, reward, state, is_terminated, action_space, state_space, AbstractEnv
using Flux
using ArgParse
using StatsBase: sample, Weights, loglikelihood, mean
using Distributions: Categorical


include("../utils/buffers.jl")
include("../utils/config_parser.jl")


Base.@kwdef struct Config
  log_frequencey::Int = 1000

  total_timesteps::Int = 50_000

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

function discounted_future_rewards(rewards, terminals, final_value, γ)
  future_rewards = zeros(eltype(rewards), size(rewards))
  future_rewards[end] = final_value  # not sure if this is correct

  # this assumes that the final state was terminal, how to fix?
  for (i, (r, t)) in enumerate(zip(reverse(rewards), reverse(terminals)))
    future_rewards[i] += t ? 0 : r + γ * future_rewards[i-1]
  end

  reverse(future_rewards)
end

function a2c()
  config = ConfigParser.argparse_struct(Config())

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
      discounted_rewards = discounted_future_rewards(data.rewards, data.terminals, values[end], config.gamma)
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
        mean(advantage .^ 2)  # TODO: this loss is *really* high can't be normal?
      end
      Flux.Optimise.update!(opt, critic_params, critic_gs)

      # reset these at end of episode
      episode_length, episode_return = 0, 0

      @info "Training Statistics" actor_loss critic_loss
    end
  end
end

@time a2c()
