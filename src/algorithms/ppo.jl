@kwdef struct PPOConfig
  total_timesteps::Int = 500_000
  batch_size::Int = 512
  num_minibatches::Int = 4
  update_epochs::Int = 4

  lr::Float64 = 2.5e-4
  gamma::Float64 = 0.99
  gae_lambda::Float64 = 0.95

  clip_coef::Float64 = 0.2
  ent_coeff::Float64 = 0.01
  v_coef::Float64 = 0.5

  normalize_advantages::Bool = true
  clip_value_loss::Bool = true
  anneal_lr::Bool = true
end

function get_action(obs::AbstractVecOrMat{Float64}, actor::Chain)
  logits = actor(obs)
  probs = Categorical.(eachcol(logits), check_args=false)
  action = rand.(probs)

  action, loglikelihood.(probs, action), entropy.(logits)
end

function logprob_actions(obs::AbstractVecOrMat{Float64}, actor::Chain, actions::AbstractVector)
  logits = actor(obs)
  probs = Categorical.(eachcol(logits), check_args=false)

  loglikelihood.(probs, actions), entropy.(logits)
end

function gae(values::AbstractVector{T}, rewards::AbstractVector{T}, terminals::AbstractVector{Bool}, γ::T, λ::T) where {T<:AbstractFloat}
  """
  Generalized advantage estimation.

  Args:
    values: [0, k]
    rewards: [1, k]
    terminals: [0,k]
    γ: gamma/discount
    λ: gae lambda

  Returns: 
   advatages [0, k-1]
  """
  advantages = similar(rewards)
  nonterm = 1.0 .- terminals

  gae = 0.0
  for t in length(rewards)-1:-1:1
    δ = rewards[t] + γ * nonterm[t+1] * values[t+1] - values[t]
    gae = δ + γ * λ * nonterm[t+1] * gae
    advantages[t] = gae
  end

  value_target = advantages + values[1:end-1]
  advantages #= , value_target =#
end

function ppo(config::PPOConfig=PPOConfig())
  nt = Threads.nthreads()
  Logger.make_logger("ppo-2-test"; to_terminal=false)

  env = MultiThreadEnv([CartPoleEnv(rng=Xoshiro(i)) for i in 1:nt])  # TODO make env configurable through CLI

  single_obs_space = single_state_space(env)
  single_act_space = single_action_space(env)
  actor, critic = Networks.make_actor_critic(single_act_space, single_obs_space)

  minibatch_size = config.batch_size ÷ config.num_minibatches
  num_updates = config.total_timesteps ÷ config.batch_size

  opt = Flux.Optimiser(ClipNorm(0.5), Adam(config.lr))  # one opt per network?

  transition = (
    state=rand(state_space(env)),
    action=rand(action_space(env)),
    logprob=ones(nt),
    reward=ones(nt),
    terminal=fill(true, nt),
    value=ones(nt),
  )

  rb = Buffer.ReplayBuffer(transition, config.batch_size)

  global_step = 0
  last_log_step = 0
  episode_returns = zeros(nt)
  episode_lengths = zeros(nt)

  start_time = time()
  reset!(env)

  next_obs = state(env)
  next_done = is_terminated(env)

  for update in 1:num_updates
    if config.anneal_lr
      frac = 1.0 - (update - 1.0) / num_updates
      opt.os[2].eta = frac * config.lr
    end

    for step in 1:config.batch_size
      global_step += nt
      episode_lengths .+= 1

      action, log_prob, entropy = get_action(next_obs, actor)
      value = critic(next_obs)

      # step env
      env(action)

      rewards = reward(env)
      Buffer.add!(rb, (
        state=next_obs,
        action=action,
        logprob=log_prob,
        reward=rewards,
        terminal=next_done,
        value=value,
      ))

      # todo: I don't like next obs - put this above
      next_obs = deepcopy(state(env))
      next_done = is_terminated(env)
      episode_returns += rewards

      if any(next_done)
        steps_per_sec = trunc(global_step / (time() - start_time))
        for i in 1:nt
          !next_done[i] && continue  # only log if terminal

          episode_return = episode_returns[i]
          episode_length = episode_lengths[i]

          # todo: would be nice if we could pass step instead of log_step_increment
          log_step_inc = last_log_step == 0 ? 0 : global_step - last_log_step

          @info "Episode Statistics" episode_return episode_length global_step steps_per_sec log_step_increment = log_step_inc
          episode_lengths[i] = 0
          episode_returns[i] = 0
          last_log_step = deepcopy(global_step)
        end

        reset!(env)
      end
    end

    # todo: this won't work because transitions are added 1 after the other
    #  need to shape replay data as (batch, num_env, ...)
    # bootstrap value if not done
    next_obs = state(env)
    next_done = is_terminated(env)
    next_values = critic(next_obs)

    advantages = gae.(
      eachrow(hcat(rb.data.value, next_values')),
      eachrow(rb.data.reward),
      eachrow(hcat(rb.data.terminal, next_done)),
      config.gamma,
      config.gae_lambda
    )
    # advantages = reduce(hcat, advantages)'
    advantages = hcat(advantages...)'
    returns = advantages + rb.data.value

    b_inds = 1:config.batch_size

    states = reshape(rb.data.state, :, config.batch_size * nt)
    actions = reshape(rb.data.action, :, config.batch_size * nt)
    logprobs = reshape(rb.data.logprob, :, config.batch_size * nt)
    values = reshape(rb.data.value, :, config.batch_size * nt)
    advantages = reshape(advantages, :, config.batch_size * nt)
    returns = reshape(returns, :, config.batch_size * nt)

    for epoch in 1:config.update_epochs
      b_inds = shuffle(b_inds)

      params = Flux.params(actor, critic)
      for start in 1:minibatch_size:config.batch_size
        loss, gs = Flux.withgradient(params) do
          end_ind = start + minibatch_size - 1
          mb_inds = b_inds[start:end_ind]

          mb_states = @view states[:, mb_inds]
          mb_actions = vec(@view actions[:, mb_inds])
          mb_advantages = @view advantages[mb_inds]
          mb_logprobs = vec(@view logprobs[mb_inds])
          mb_values = @view values[mb_inds]
          mb_returns = @view returns[mb_inds]

          newlogprob, entropy = logprob_actions(mb_states, actor, mb_actions)
          newvalue = critic(mb_states)
          newlogprob = vec(newlogprob)
          newvalue = vec(newvalue)

          # policy loss
          mb_advantages = if config.normalize_advantages
            # todo: revisit fused vector ops in julia perf tips
            (mb_advantages .- mean(mb_advantages)) ./ (std(mb_advantages) .+ 1e-8)
          end

          logratio = newlogprob - mb_logprobs
          ratio = exp.(logratio)
          pg_loss1 = @. -mb_advantages * ratio
          pg_loss2 = @. -mb_advantages * clamp.(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
          pg_loss = mean(max.(pg_loss1, pg_loss2))

          # value loss
          v_loss = if config.clip_value_loss
            v_loss_unclipped = mean(newvalue .- mb_returns .^ 2)
            # todo: revisit fused vector ops in julia perf tips
            v_clipped = @. mb_values + clamp(newvalue - mb_values, -config.clip_coef, config.clip_coef)
            v_loss_clipped = @. (v_clipped - mb_returns)^2
            v_loss_max = max.(v_loss_unclipped, v_loss_clipped)
            0.5 * mean(v_loss_max)
          else
            0.5 * mean((newvalue - mb_returns) .^ 2)
          end

          entropy = mean(entropy)
          pg_loss - config.ent_coeff * entropy + config.v_coef * v_loss
        end

        log_step_inc = last_log_step == 0 ? 0 : global_step - last_log_step
        # todo: log loss components
        @info "Training Statistics" loss log_step_increment = log_step_inc
        last_log_step = deepcopy(global_step)
        Flux.Optimise.update!(opt, params, gs)
      end
    end
    Buffer.clear!(rb)
  end
end
