using ReinforcementLearningEnvironments: MountainCarEnv
using ReinforcementLearningBase: reset!, reward, state, is_terminated, action_space, state_space, AbstractEnv
using Flux
using StatsBase: mean

using Dates: now, format

include("../utils/buffers.jl")
include("../utils/config_parser.jl")
include("../utils/logger.jl")


Base.@kwdef struct Config
  run_name::String = format(now(), "yy-mm-dd|HH:MM:SS")

  log_frequencey::Int = 1000

  total_timesteps::Int = 50_000

  buffer_size::Int64 = 10_000
  min_buff_size::Int64 = 200

  train_freq::Int64 = 10
  target_net_freq::Int64 = 100
  batch_size::Int64 = 120
  gamma::Float64 = 0.99
end

function make_nn(env::AbstractEnv)
  ob_size = length(state_space(env).s)
  ac_size = length(action_space(env).s)

  actor = Chain(Dense(ob_size, 64, relu), Dense(64, 64, relu), Dense(84, ac_size))
  critic = Chain(Dense(ob_size + ac_size, 64, relu), Dense(64, 64, relu), Dense(84, 1))

  actor, critic
end

get_q(critic, obs, act) = critic(hcat(obs, act))

function dqn()
  config = ConfigParser.argparse_struct(Config())
  Logger.make_logger(config.run_name)

  env = MountainCarEnv(continuous=true)  # TODO make env configurable through CLI

  actor, critic = make_nn(env)
  target_actor = deepcopy(actor)
  target_critic = deepcopy(critic)
  opt = ADAM()

  rb = Buffers.ReplayBuffer(config.buffer_size)

  episode_return = 0
  episode_length = 0

  start_time = time()
  reset!(env)
  for global_step in 1:config.total_timesteps
    obs = deepcopy(state(env))  # state needs to be coppied otherwise state and next_state is the same
    # action selection
    action = actor(obs)
    env(action)  # step env

    # add to buffer
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
      @info "Episode Statistics" episode_return episode_length
      episode_length, episode_return = 0, 0

      reset!(env)
    end

    # Learning
    if (global_step > config.min_buff_size) && (global_step % config.train_freq == 0)
      data = Buffers.sample(rb, config.batch_size)

      next_acts = target_actor(data.next_states)
      next_q = get_q(target_critic, data.next_states, next_acts)
      td_target = data.rewards + config.gamma * next_q .* (1.0 .- data.terminals)

      # critic update
      critic_params = Flux.params(critic)
      loss, gs = Flux.withgradient(critic_params) do
        q = get_q(critic, data.actions, data.states)
        Flux.mse(td_target, q)
      end
      Flux.Optimise.update!(opt, critic_params, gs)

      # actor update
      actor_params = Flux.params(actor)
      loss, gs = Flux.withgradient(actor_params) do
        -mean(get_q(critic, data.states, actor(data.states)))
      end
      Flux.Optimise.update!(opt, actor_params, gs)

      # TODO slowly move towards target nets
      # target_critic = critic * 1

      if global_step % config.log_frequencey == 0
        steps_per_second = trunc(global_step / (time() - start_time))
        @info "Training Statistics" loss steps_per_second
      end
    end
  end
end

@time dqn()
