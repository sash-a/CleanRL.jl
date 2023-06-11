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
  in_size = length(single_state_space(env))
  out_size = length(single_action_space(env))
  Chain(Dense(in_size, 120, relu), Dense(120, 84, relu), Dense(84, out_size))
end

function linear_schedule(start_ϵ, end_ϵ, duration, t)
  slope = (end_ϵ - start_ϵ) / duration
  max(slope * t + start_ϵ, end_ϵ)
end


function dqn(config::DQNConfig=DQNConfig())
  nt = Threads.nthreads()
  Logger.make_logger("dqn|$(config.run_name)"; to_terminal=false)

  env = MultiThreadEnv([CartPoleEnv(rng=Xoshiro(i)) for i in 1:nt])  # TODO make env configurable through CLI

  q_net = make_nn(env)
  target_net = deepcopy(q_net)
  opt = Adam(config.lr)

  transition = (
    state=rand(single_state_space(env)),
    action=rand(single_action_space(env)),
    reward=1.0,
    next_state=rand(single_state_space(env)),
    terminal=true
  )
  rb = Buffer.ReplayBuffer(transition, config.buffer_size)

  ϵ_schedule = t -> linear_schedule(config.epsilon_start, config.epsilon_end, config.epsilon_duration, t)

  episode_returns = zeros(nt)
  episode_lengths = zeros(nt)
  last_log_step = 0

  start_time = time()
  reset!(env; is_force=true)
  for n_iters in 1:config.total_timesteps÷nt
    obs = deepcopy(state(env))  # state needs to be coppied otherwise state and next_state is the same

    # action selection
    ϵ = ϵ_schedule(nt * n_iters)
    explore = rand() < ϵ
    action = explore ? rand(action_space(env)) : argmax.(eachcol(q_net(obs)))

    env(action)  # step env

    # add to buffer
    rewards = reward(env)
    terminals = is_terminated(env)
    @inbounds for i in 1:nt
      # splitting up the batched transition so each row in rb is a single transition
      transition = (
        state=obs[:, i],
        action=action[i, :],
        reward=rewards[i, :],
        next_state=deepcopy(state(env)[:, i]),
        terminal=terminals[i, :]
      )
      Buffer.add!(rb, transition)
    end

    episode_returns += rewards
    episode_lengths += ones(nt)
    if any(terminals)
      steps_per_sec = nt * trunc(n_iters / (time() - start_time))
      for i in 1:nt
        !terminals[i] && continue  # only log if terminal

        episode_return = episode_returns[i]
        episode_length = episode_lengths[i]

        # todo: would be nice if we could pass step instead of log_step_increment
        log_step_inc = last_log_step == 0 ? 0 : n_iters * nt - last_log_step

        @info "Episode Statistics" episode_return episode_length n_iters ϵ steps_per_sec log_step_increment = log_step_inc
        episode_lengths[i] = 0
        episode_returns[i] = 0
        last_log_step = n_iters * nt
      end

      reset!(env)
    end

    # Learning
    if (n_iters > config.min_buff_size) && (n_iters % config.train_freq * nt == 0)
      data = Buffer.sample(rb, config.batch_size)
      # Convert actions to CartesianIndexes so they can be used to index q matrix
      actions = CartesianIndex.(vec(data.action), 1:length(data.action))

      next_q = data.next_state |> target_net |> eachcol .|> maximum
      td_target = vec(data.reward) + config.gamma * next_q .* (1.0 .- vec(data.terminal))

      # Get grads and update model
      params = Flux.params(q_net)
      loss, gs = Flux.withgradient(params) do
        q = q_net(data.state)
        q = q[actions]
        Flux.mse(td_target, q)
      end
      Flux.Optimise.update!(opt, params, gs)

      if n_iters % config.target_net_freq == 0
        target_net = deepcopy(q_net)
      end

      if n_iters % config.log_frequencey == 0
        # todo: would be nice if we could pass step instead of log_step_increment
        log_step_inc = last_log_step == 0 ? 0 : n_iters * nt - last_log_step
        @info "Training Statistics" loss log_step_increment = log_step_inc
        last_log_step = n_iters * nt
      end
    end
  end
end

