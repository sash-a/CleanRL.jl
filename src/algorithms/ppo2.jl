using ReinforcementLearningEnvironments: CartPoleEnv
using ReinforcementLearningBase: reset!, reward, state, is_terminated, action_space, state_space, AbstractEnv

using Flux
using StatsBase: sample, Weights, loglikelihood, mean, entropy, std
using Random: shuffle
using Distributions: Categorical
using ChainRulesCore

using Dates: now, format

include("../utils/replay_buffer.jl")
include("../utils/config_parser.jl")
include("../utils/logger.jl")
include("../utils/networks.jl")

@kwdef struct Config
  batch_size::Int = 512
  total_timesteps::Int = 500_000
  num_minibatches::Int = 4
  update_epochs::Int = 4
  lr::Float64 = 2.5e-4
  gamma::Float64 = 0.99
  gae_lambda::Float64 = 0.95
  clip_coef::Float64 = 0.2
  ent_coeff::Float64 = 0.01
  v_coef::Float64 = 0.5
end

function get_action_and_value(state::AbstractVecOrMat{Float64}, actor::Chain, critic::Chain, action::Union{VecOrMat{Int64},Nothing}=nothing)
  logits = actor(state)
  # the collect here is slow, but check_args=false is giving worse results :(
  probs = Categorical.(collect.(eachcol(logits)))
  if action === nothing
    action = rand.(probs)
  end

  lp = loglikelihood.(probs, action)

  action, lp, entropy.(logits), critic(state)
end

function ppo(config::Config=Config())
  Logger.make_logger("ppo-2-test")

  env = CartPoleEnv()  # TODO make env configurable through argparse

  actor = Chain(
    Dense(4, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 2),
    softmax,
  )
  critic = Chain(
    Dense(4, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 1)
  )

  minibatch_size = config.batch_size ÷ config.num_minibatches
  num_updates = config.total_timesteps ÷ config.batch_size

  opt = Flux.Optimiser(ClipNorm(0.5), Adam(2.5e-4, (0.9, 0.999), 1e-5))  # one opt per network?

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

  start_time = time()
  reset!(env)

  next_obs = state(env)
  next_done = is_terminated(env)

  for update in 1:num_updates
    frac = 1.0 - (update - 1.0) / num_updates
    opt.os[2].eta = frac * config.lr
    for step in 1:config.batch_size
      global_step += 1

      action, log_prob, entropy, value = get_action_and_value(next_obs, actor, critic)

      # step env
      env(action[1])  # todo cartpole expects an int, but other envs may expect a vec

      Buffer.add!(rb, (
        state=deepcopy(next_obs),
        action=deepcopy(action),
        logprob=log_prob,
        reward=deepcopy([reward(env)]),
        terminal=[next_done],
        value=value,
      ))
      next_obs = deepcopy(state(env))
      next_done = deepcopy(is_terminated(env))
      episode_return += last(rb).reward[1]

      if next_done
        reset!(env)
        next_obs = state(env)
        next_done = is_terminated(env)
        steps_per_sec = global_step / (time() - start_time)
        @info "Episode Statistics" episode_return global_step steps_per_sec step
        episode_return = 0
        episode_length = 0
      end
    end

    # bootstrap value if not done
    next_value = critic(next_obs)[1]
    advantages = similar(rb.data.reward)
    lastgaelam = 0
    for t in reverse(1:config.batch_size-1)
      if t == config.batch_size - 1
        nextnonterminal = 1.0 - next_done
        nextvalue = next_value
      else
        nextnonterminal = 1.0 - rb.data.terminal[t+1]
        nextvalue = rb.data.value[t+1]
      end
      delta = rb.data.reward[t] + config.gamma * nextvalue * nextnonterminal - rb.data.value[t]
      advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
    end
    returns = advantages + rb.data.value

    b_inds = 1:config.batch_size
    # do I need to copy buffer here?

    for epoch in 1:config.update_epochs
      b_inds = shuffle(b_inds)

      params = Flux.params(actor, critic)
      for start in 1:minibatch_size:config.batch_size
        loss, gs = Flux.withgradient(params) do
          end_ind = start + minibatch_size - 1
          mb_inds = b_inds[start:end_ind]

          _, newlogprob, entropy, newvalue = get_action_and_value(rb.data.state[mb_inds, :]', actor, critic, rb.data.action[mb_inds, :])
          logratio = dropdims(newlogprob; dims=2) - rb.data.logprob[mb_inds]
          ratio = exp.(logratio)

          # todo: norm adv?
          # policy loss
          mb_advantages = (advantages[mb_inds] .- mean(advantages[mb_inds])) / (std(advantages[mb_inds]) .+ 1e-8)
          pg_loss1 = -mb_advantages .* ratio
          pg_loss2 = -mb_advantages .* clamp.(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
          pg_loss = mean(max.(pg_loss1, pg_loss2))

          # value loss
          # todo: optional clip
          # v_loss = 0.5 * mean((dropdims(newvalue; dims=1) - returns[mb_inds]) .^ 2)
          newvalue = dropdims(newvalue; dims=1)
          v_loss_unclipped = mean(newvalue - returns[mb_inds] .^ 2)
          v_clipped = rb.data.value[mb_inds] + clamp.(newvalue - rb.data.value[mb_inds], -config.clip_coef, config.clip_coef)
          v_loss_clipped = (v_clipped - returns[mb_inds]) .^ 2
          v_loss_max = max.(v_loss_unclipped, v_loss_clipped)
          v_loss = 0.5 * mean(v_loss_max)

          pg_loss - config.ent_coeff * mean(entropy) + config.v_coef * v_loss
        end

        @info "Training Statistics" loss
        Flux.Optimise.update!(opt, params, gs)
      end
    end
    Buffer.clear!(rb)
  end
end

# ppo()
