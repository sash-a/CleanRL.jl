using ReinforcementLearningEnvironments: CartPoleEnv
using ReinforcementLearningBase: reset!, reward, state, action_space, is_terminated
using Flux
using ArgParse

include("../utils/buffers.jl")
include("../utils/config_parser.jl")


Base.@kwdef struct Config
  total_timesteps::Int = 50_000
  buffer_size::Int64 = 10_000
  min_buff_size::Int64 = 200
  train_freq::Int64 = 10
  target_net_freq::Int64 = 100
  batch_size::Int64 = 120
  γ::Float64 = 0.99
end

function linear_schedule(start_ϵ, end_ϵ, duration, t)
  slope = (end_ϵ - start_ϵ) / duration
  max(slope * t + start_ϵ, end_ϵ)
end


function dqn()
  config = ConfigParser.argparse_struct(Config())

  env = CartPoleEnv()  # TODO make env configurable through argparse
  # TODO: make layer size depend on env
  q_net = Chain(Dense(4, 120, relu), Dense(120, 84, relu), Dense(84, 2))
  opt = ADAM()
  target_net = deepcopy(q_net)

  rb = Buffers.ReplayBuffer(config.buffer_size)
  ϵ_schedule = t -> linear_schedule(1.0, 0.05, 10_000, t)
  ep_rew = 0

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

    ep_rew += reward(env)
    if is_terminated(env)
      @show ep_rew
      ep_rew = 0
      reset!(env)
    end


    if (global_step > config.min_buff_size) && (global_step % config.train_freq == 0)
      data = Buffers.sample(rb, config.batch_size)
      actions = CartesianIndex.(data.actions, 1:length(data.actions))

      next_q = data.next_states |> target_net |> eachcol .|> maximum
      td_target = data.rewards + config.γ * next_q .* (1.0 .- data.terminals)

      params = Flux.params(q_net)
      gs = gradient(params) do
        q = data.states |> q_net
        q = q[actions]
        Flux.mse(td_target, q)
      end

      Flux.Optimise.update!(opt, params, gs)

      if global_step % config.target_net_freq == 0
        target_net = deepcopy(q_net)
      end
    end
  end
end

dqn()
