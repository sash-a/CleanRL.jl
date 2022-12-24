using ReinforcementLearningEnvironments: CartPoleEnv
using ReinforcementLearningBase: reset!, reward, state, is_terminated, action_space, state_space, AbstractEnv
using Flux
using ArgParse
using StatsBase: sample, Weights, loglikelihood
using Distributions: Categorical


include("../utils/buffers.jl")
include("../utils/config_parser.jl")


Base.@kwdef struct Config
  log_frequencey::Int = 1000

  total_timesteps::Int = 50_000

  batch_size::Int64 = 120
  gamma::Float64 = 0.99
end

function make_nn(env::AbstractEnv)
  in_size = length(state_space(env).s)
  out_size = length(action_space(env).s)
  ob_net = Chain(Dense(in_size, 120, relu), Dense(120, 84, relu))
  actor = Chain(ob_net, Dense(84, out_size), softmax)
  critic = Chain(ob_net, Dense(84, 1))

  actor, critic
end

function a2c()
  config = ConfigParser.argparse_struct(Config())

  env = CartPoleEnv()  # TODO make env configurable through argparse

  actor, critic = make_nn(env)
  opt = ADAM()

  rb = Buffers.ReplayQueue{Buffers.PGTransition}()

  episode_return = 0
  episode_length = 0

  start_time = time()
  reset!(env)
  for global_step in 1:config.total_timesteps
    obs = deepcopy(state(env))  # state needs to be coppied otherwise state and next_state is the same

    ac_dist = Categorical(actor(obs))
    action = rand(ac_dist)
    logprob = loglikelihood(ac_dist, action)

    env(action)

    transition = Buffers.PGTransition(
      obs,
      logprob,
      reward(env),
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

      # learn
      data = sample(rb, episode_length)
      values = critic(data.states)
      discounted_rewards = discount_rewards(data.rewards)
      advantage = discounted_rewards - values

      # actor update
      actor_params = Flux.params(actor)
      actor_gs = Flux.gradient(actor_params) do
        data.logprob * advantage
      end
      Flux.Optimiser.update!(opt, actor_params, actor_gs)

      # critic update
      critic_params = Flux.params(critic)
      critic_gs = Flux.gradient(critic_params) do
        advantage^2
      end
      Flux.Optimiser.update(opt, critic_params, critic_gs)
    end

    # # Learning
    # if (global_step > config.min_buff_size) && (global_step % config.train_freq == 0)
    #   data = Buffers.sample(rb, config.batch_size)
    #   # Convert actions to CartesianIndexes so they can be used to index matricies
    #   actions = CartesianIndex.(data.actions, 1:length(data.actions))
    #
    #   next_q = data.next_states |> target_net |> eachcol .|> maximum
    #   td_target = data.rewards + config.gamma * next_q .* (1.0 .- data.terminals)
    #
    #   # Get grads and update model
    #   params = Flux.params(q_net)
    #   loss, gs = Flux.withgradient(params) do
    #     q = data.states |> q_net
    #     q = q[actions]
    #     Flux.mse(td_target, q)
    #   end
    #   Flux.Optimise.update!(opt, params, gs)
    #
    #   if global_step % config.target_net_freq == 0
    #     target_net = deepcopy(q_net)
    #   end
    #
    #   if global_step % config.log_frequencey == 0
    #     steps_per_second = trunc(global_step / (time() - start_time))
    #     @info "Training statistics" loss steps_per_second
    #   end
    # end
  end
end

@time a2c()
