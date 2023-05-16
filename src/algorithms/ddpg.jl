Base.@kwdef struct DDPGConfig
  run_name::String = format(now(), "yy-mm-dd|HH:MM:SS")

  log_frequencey::Int = 1000

  total_timesteps::Int = 5_000_000

  buffer_size::Int64 = 10_000
  min_buff_size::Int64 = 200

  lr::Float64 = 0.0001
  train_freq::Int64 = 10
  target_net_freq::Int64 = 100
  tau::Float64 = 0.005

  batch_size::Int64 = 120
  gamma::Float64 = 0.99
end

# todo: how to work with ClosedInterval to get correct action shape
function make_ddpg_nn(env::AbstractEnv)
  ob_size = length(state_space(env))
  ac_size = 1  # length(action_space(env).s)

  actor = Chain(
    Dense(ob_size, 64, tanh_fast),
    Dense(64, 64, tanh_fast),
    Dense(64, ac_size, tanh_fast),
    x -> x * 2
  )
  critic = Chain(
    Dense(ob_size + ac_size, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 1)
  )

  actor, critic
end

function polyak_update!(target::Zygote.Params, source::Zygote.Params, tau::Float64)
  for (t, s) in zip(target, source)
    t .= tau .* s .+ (1.0 - tau) .* t
  end
end

get_q(critic, obs, act) = critic(vcat(obs, act))

function ddpg(config::DDPGConfig=DDPGConfig())
  Logger.make_logger("ddpg|$(config.run_name)")

  env = PendulumEnv(continuous=true)  # TODO make env configurable through CLI

  actor, critic = make_ddpg_nn(env)
  target_actor = deepcopy(actor)
  target_critic = deepcopy(critic)
  actor_opt = Adam(config.lr)
  critic_opt = Adam(config.lr)

  transition = (
    state=rand(state_space(env)),
    action=rand(action_space(env)),
    reward=1.0,
    next_state=rand(state_space(env)),
    terminal=true
  )

  rb = Buffer.ReplayBuffer(transition, config.buffer_size)

  episode_return = 0
  episode_length = 0

  start_time = time()
  reset!(env)
  for global_step in 1:config.total_timesteps
    obs = deepcopy(state(env))  # state needs to be coppied otherwise state and next_state is the same
    action = actor(obs)  # action selection
    env(action[1])  # step env

    # add to buffer
    transition = (
      state=obs,
      action=action,
      reward=[reward(env)],
      next_state=deepcopy(state(env)),
      terminal=[is_terminated(env)]
    )
    Buffer.add!(rb, transition)

    # Recording episode statistics
    episode_return += transition.reward[1]
    episode_length += 1
    if transition.terminal[1]
      steps_per_second = trunc(global_step / (time() - start_time))
      @info "Episode Statistics" episode_return episode_length steps_per_second
      episode_length, episode_return = 0, 0

      reset!(env)
    end

    # Learning
    # todo too many transposes
    if (global_step > config.min_buff_size) && (global_step % config.train_freq == 0)
      data = Buffer.sample(rb, config.batch_size)

      next_acts = target_actor(data.next_state')
      next_q = get_q(target_critic, data.next_state', next_acts)'
      td_target = data.reward + config.gamma * next_q .* (1.0 .- data.terminal)

      # critic update
      critic_params = Flux.params(critic)
      critic_loss, gs = Flux.withgradient(critic_params) do
        q = get_q(critic, data.state', data.action')'
        Flux.mse(td_target, q)
      end
      Flux.Optimise.update!(critic_opt, critic_params, gs)

      # actor update
      actor_params = Flux.params(actor)
      actor_loss, gs = Flux.withgradient(actor_params) do
        -mean(get_q(critic, data.state', actor(data.state')))
      end
      Flux.Optimise.update!(actor_opt, actor_params, gs)

      # Move towards target nets
      polyak_update!(Flux.params(target_actor), Flux.params(actor), config.tau)
      polyak_update!(Flux.params(target_critic), Flux.params(critic), config.tau)

      if global_step % config.log_frequencey == 0
        @info "Training Statistics" actor_loss critic_loss
      end
    end
  end
end

