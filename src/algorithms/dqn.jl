Base.@kwdef struct DQNConfig
  run_name::String = format(now(), "yy-mm-dd|HH:MM:SS")

  log_frequencey::Int = 1000

  total_timesteps::Int = 500_000

  buffer_size::Int64 = 10_000
  min_buff_size::Int64 = 200

  lr::Float64 = 0.0001
  train_freq::Int64 = 10
  target_net_freq::Int64 = 100
  batch_size::Int64 = 120
  gamma::Float64 = 0.99

  epsilon_start::Float64 = 1.0
  epsilon_end::Float64 = 0.05
  epsilon_duration::Float64 = 10_000
end

function make_nn(env::AbstractEnv)
  in_size = length(state_space(env))
  out_size = length(action_space(env))
  Chain(Dense(in_size, 120, relu), Dense(120, 84, relu), Dense(84, out_size))
end

function linear_schedule(start_ϵ, end_ϵ, duration, t)
  slope = (end_ϵ - start_ϵ) / duration
  max(slope * t + start_ϵ, end_ϵ)
end


function dqn(config::DQNConfig=DQNConfig())
  Logger.make_logger(config.run_name)

  env = CartPoleEnv()  # TODO make env configurable through CLI

  q_net = make_nn(env)
  target_net = deepcopy(q_net)
  opt = Adam(config.lr)

  transition = (
    state=rand(state_space(env)),
    action=rand(action_space(env)),
    reward=1.0,
    next_state=rand(state_space(env)),
    terminal=true
  )
  rb = Buffer.ReplayBuffer(transition, config.buffer_size)

  ϵ_schedule = t -> linear_schedule(config.epsilon_start, config.epsilon_end, config.epsilon_duration, t)

  episode_return = 0
  episode_length = 0

  start_time = time()
  reset!(env)
  for global_step in 1:config.total_timesteps
    obs = deepcopy(state(env))  # state needs to be coppied otherwise state and next_state is the same
    # action selection
    ϵ = ϵ_schedule(global_step)
    action = if rand() < ϵ
      env |> action_space |> rand
    else
      qs = q_net(obs)
      argmax(qs)
    end

    env(action)  # step env

    # add to buffer
    transition = (
      state=obs,
      action=[action],
      reward=[reward(env)],
      next_state=deepcopy(state(env)),
      terminal=[is_terminated(env)]
    )
    Buffer.add!(rb, transition)

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
      data = Buffer.sample(rb, config.batch_size)
      # Convert actions to CartesianIndexes so they can be used to index q matrix
      actions = CartesianIndex.(data.action, 1:length(data.action))

      next_q = data.next_state' |> target_net |> eachcol .|> maximum
      td_target = data.reward + config.gamma * next_q .* (1.0 .- data.terminal)

      # Get grads and update model
      params = Flux.params(q_net)
      loss, gs = Flux.withgradient(params) do
        q = data.state' |> q_net
        q = q[actions]
        Flux.mse(td_target, q)
      end
      Flux.Optimise.update!(opt, params, gs)

      if global_step % config.target_net_freq == 0
        target_net = deepcopy(q_net)
      end

      if global_step % config.log_frequencey == 0
        steps_per_second = trunc(global_step / (time() - start_time))
        @info "Training Statistics" loss steps_per_second
      end
    end
  end
end

