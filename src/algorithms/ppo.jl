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

function action_and_value(state::AbstractVecOrMat{Float64}, actor::Chain, critic::Chain, action::Union{AbstractVector,Nothing}=nothing)
  logits = actor(state)  # note these are sofmaxed in the network
  probs = Categorical.(eachcol(logits), check_args=false)

  if action === nothing
    action = rand.(probs)
  end

  # it's likely faster to do these calculations manually: softmax, logsoftmax[action]
  action, logpdf.(probs, action), entropy.(logits), critic(state)
end

function gae(values::Vector{T}, rewards::Vector{T}, terminals::Vector{Bool}, γ::T, λ::T) where {T<:AbstractFloat}
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
  advantages, value_target
end

function ppo(config::PPOConfig=PPOConfig())
  Logger.make_logger("ppo-2-test"; to_terminal=false)

  env = CartPoleEnv(max_steps=500)  # TODO make env configurable through argparse

  actor, critic = Networks.make_actor_critic(env)

  minibatch_size = config.batch_size ÷ config.num_minibatches
  num_updates = config.total_timesteps ÷ config.batch_size

  opt = Flux.Optimiser(ClipNorm(0.5), Adam(config.lr))  # one opt per network?

  transition = (
    state=rand(state_space(env)),
    action=rand(action_space(env)),
    logprob=1.0,
    reward=1.0,
    terminal=true,
    value=1.0
  )

  rb = Buffer.ReplayBuffer(transition, config.batch_size)

  global_step = 0
  episode_return = 0
  episode_length = 0
  last_log_step = 0

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
      global_step += 1
      episode_length += 1

      action, log_prob, entropy, value = action_and_value(next_obs, actor, critic)

      # step env
      env(action[1])  # todo cartpole expects an int, but other envs may expect a vec

      Buffer.add!(rb, (
        state=next_obs,
        action=action,
        logprob=log_prob,
        reward=[reward(env)],
        terminal=[next_done],
        value=value,
      ))
      next_obs = deepcopy(state(env))
      next_done = is_terminated(env)
      episode_return += last(rb).reward[1]

      if next_done
        reset!(env)
        steps_per_sec = global_step / (time() - start_time)

        # todo: would be nice if we could pass step instead of log_step_increment
        log_step_inc = last_log_step == 0 ? 0 : global_step - last_log_step
        @info "Episode Statistics" episode_return episode_length global_step steps_per_sec log_step_increment = log_step_inc
        last_log_step = global_step

        episode_return = 0
        episode_length = 0
      end
    end

    # bootstrap value if not done
    next_value = critic(next_obs)[1]
    advantages, returns = gae(
      vcat(vec(rb.data.value), next_value),
      vec(rb.data.reward),
      vcat(vec(rb.data.terminal), next_done),
      config.gamma,
      config.gae_lambda
    )

    b_inds = 1:config.batch_size

    for epoch in 1:config.update_epochs
      b_inds = shuffle(b_inds)

      params = Flux.params(actor, critic)
      for start in 1:minibatch_size:config.batch_size
        pg_loss = 0.0
        v_loss = 0.0
        entropy_loss = 0.0

        loss, gs = Flux.withgradient(params) do
          end_ind = start + minibatch_size - 1
          mb_inds = b_inds[start:end_ind]

          mb_states = @view rb.data.state[:, mb_inds]
          mb_actions = vec(@view rb.data.action[:, mb_inds])
          mb_advantages = @view advantages[mb_inds]
          mb_logprobs = vec(@view rb.data.logprob[mb_inds])
          mb_values = @view rb.data.value[mb_inds]
          mb_returns = @view returns[mb_inds]

          _, newlogprob, entropy, newvalue = action_and_value(mb_states, actor, critic, mb_actions)
          newlogprob = vec(newlogprob)
          newvalue = vec(newvalue)

          # policy loss
          mb_advantages = if config.normalize_advantages
            (mb_advantages .- mean(mb_advantages)) / (std(mb_advantages) .+ 1e-8)
          end

          logratio = newlogprob - mb_logprobs
          ratio = exp.(logratio)
          pg_loss1 = -mb_advantages .* ratio
          pg_loss2 = -mb_advantages .* clamp.(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
          pg_loss = mean(max.(pg_loss1, pg_loss2))

          # value loss
          v_loss = if config.clip_value_loss
            v_loss_unclipped = mean(newvalue - mb_returns .^ 2)
            v_clipped = mb_values + clamp.(newvalue - mb_values, -config.clip_coef, config.clip_coef)
            v_loss_clipped = (v_clipped - mb_returns) .^ 2
            v_loss_max = max.(v_loss_unclipped, v_loss_clipped)
            0.5 * mean(v_loss_max)
          else
            0.5 * mean((newvalue - mb_returns) .^ 2)
          end

          entropy_loss = mean(entropy)
          pg_loss - config.ent_coeff * entropy_loss + config.v_coef * v_loss
        end

        # todo: log loss components
        log_step_inc = last_log_step == 0 ? 0 : global_step - last_log_step
        @info "Training Statistics" loss pg_loss v_loss entropy_loss log_step_increment = log_step_inc
        last_log_step = deepcopy(global_step)

        Flux.Optimise.update!(opt, params, gs)
      end
    end
    # Buffer.clear!(rb)
  end
end
