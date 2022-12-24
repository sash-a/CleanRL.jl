using ReinforcementLearningEnvironments: CartPoleEnv
using ReinforcementLearningBase: reset!, reward, state, is_terminated, action_space, state_space, AbstractEnv
using Flux
using ArgParse

using ReinforcementLearningBase

include("../utils/buffers.jl")
include("../utils/config_parser.jl")


Base.@kwdef struct Config
  log_frequencey::Int = 1000

  total_timesteps::Int = 50_000

  buffer_size::Int64 = 10_000
  min_buff_size::Int64 = 200

  train_freq::Int64 = 10
  target_net_freq::Int64 = 100
  batch_size::Int64 = 120
  gamma::Float64 = 0.99

  epsilon_start::Float64 = 1.0
  epsilon_end::Float64 = 0.05
  epsilon_duration::Float64 = 10_000
end

function make_nn(env::AbstractEnv)
  in_size = length(state_space(env).s)
  out_size = length(action_space(env).s)
  Chain(Dense(in_size, 120, relu), Dense(120, 84, relu), Dense(84, out_size))
end

function linear_schedule(start_ϵ, end_ϵ, duration, t)
  slope = (end_ϵ - start_ϵ) / duration
  max(slope * t + start_ϵ, end_ϵ)
end


function dqn()
  config = ConfigParser.argparse_struct(Config())

  env = CartPoleEnv()  # TODO make env configurable through argparse

  q_net = make_nn(env)
  target_net = deepcopy(q_net)
  opt = ADAM()

  rb = Buffers.ReplayBuffer(config.buffer_size)
  ϵ_schedule = t -> linear_schedule(config.epsilon_start, config.epsilon_end, config.epsilon_duration, t)

  episode_return = 0
  episode_length = 0

  start_time = time()
  reset!(env)
  for global_step in 1:config.total_timesteps
    obs = deepcopy(state(env))  # state needs to be coppied otherwise state and next_state is the same

    ϵ = ϵ_schedule(global_step)
    action = if rand() < ϵ
      env |> action_space |> rand
    else
      qs = q_net(obs)
      argmax(qs)
    end

    env(action)

    transition = Buffers.Transition(
      obs,
      action,
      reward(env),
      deepcopy(state(env)),
      is_terminated(env)
    )
    Buffers.add!(rb, transition)

    # Recording episode statistics
    episode_return += reward(env)
    episode_length += 1
    if is_terminated(env)
      @info "Episode Statistics" episode_return episode_length ϵ
      episode_length, episode_return = 0, 0

      reset!(env)
    end

    # Learning
    if (global_step > config.min_buff_size) && (global_step % config.train_freq == 0)
      data = Buffers.sample(rb, config.batch_size)
      # Convert actions to CartesianIndexes so they can be used to index matricies
      actions = CartesianIndex.(data.actions, 1:length(data.actions))

      next_q = data.next_states |> target_net |> eachcol .|> maximum
      td_target = data.rewards + config.gamma * next_q .* (1.0 .- data.terminals)

      # Get grads and update model
      params = Flux.params(q_net)
      loss, gs = Flux.withgradient(params) do
        q = data.states |> q_net
        q = q[actions]
        Flux.mse(td_target, q)
      end
      Flux.Optimise.update!(opt, params, gs)

      if global_step % config.target_net_freq == 0
        target_net = deepcopy(q_net)
      end

      if global_step % config.log_frequencey == 0
        steps_per_second = trunc(global_step / (time() - start_time))
        @info "Training statistics" loss steps_per_second
      end
    end
  end
end

@time dqn()
