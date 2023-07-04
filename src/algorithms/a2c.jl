Base.@kwdef struct A2CConfig
  run_name::String = format(now(), "yy-mm-dd|HH:MM:SS")

  lr::Float64 = 0.0001

  total_timesteps::Int = 1_000_000
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
function a2c(config::A2CConfig=A2CConfig())
  Logger.make_logger("a2c|$(config.run_name)")

  env = CartPoleEnv(max_steps=500)  # TODO make env configurable through argparse

  actor, critic = Networks.make_actor_critic(env)
  opt = Flux.Optimiser(ClipNorm(0.5), Adam(config.lr))

  transition = (
    state=rand(state_space(env)),
    action=rand(action_space(env)),
    reward=1.0,
    terminal=true
  )
  rb = Buffer.ReplayBuffer(transition, config.min_replay_size * 2)

  episode_return = 0
  episode_length = 0

  start_time = time()
  reset!(env)
  for global_step in 1:config.total_timesteps
    # state needs to be coppied otherwise each loop will overwrite ref in replay buffer
    obs = deepcopy(state(env))
    probs = softmax(actor(obs))
    ac_dist = Categorical(probs)
    action = rand(ac_dist)

    env(action)  # step env

    transition = (
      state=obs,
      action=[action],
      reward=[reward(env)],
      terminal=[is_terminated(env)]
    )
    Buffer.add!(rb, transition)

    # Recording episode statistics
    episode_return += transition.reward[1]
    episode_length += 1

    if is_terminated(env)  # todo: might be missing a final transition
      if rb.size > config.min_replay_size  # training
        data = map(x -> x[:, 1:rb.size], rb.data) # todo -1 or not here?
        final_value = critic(state(env))[1]
        discounted_rewards = discounted_future_rewards(vec(data.reward), vec(data.terminal), final_value, config.gamma)

        # critic update
        advantage = []
        critic_params = Flux.params(critic)
        critic_loss, critic_gs = Flux.withgradient(critic_params) do
          values = critic(data.state)
          advantage = discounted_rewards - vec(values)
          mean(advantage .^ 2)
        end
        Flux.Optimise.update!(opt, critic_params, critic_gs)

        # actor update
        actor_params = Flux.params(actor)
        actor_loss, actor_gs = Flux.withgradient(actor_params) do
          ac_probs = data.state |> actor |> softmax
          ac_dists = Categorical.(eachcol(ac_probs), check_args=false)
          # todo: grad of logpdf is slow, manually do logsoftmax.
          log_probs = logpdf.(ac_dists, vec(data.action))
          -mean(log_probs .* advantage)
        end
        Flux.Optimise.update!(opt, actor_params, actor_gs)

        # logging
        @info "Training Statistics" actor_loss critic_loss

        Buffer.clear!(rb)
      end

      steps_per_sec = trunc(global_step / (time() - start_time))
      @info "Episode Statistics" episode_return episode_length global_step steps_per_sec
      # reset counters
      episode_length, episode_return = 0, 0
      reset!(env)
    end

  end
end

