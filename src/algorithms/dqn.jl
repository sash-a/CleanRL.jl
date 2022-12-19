using ReinforcementLearningEnvironments: CartPoleEnv
using ReinforcementLearningBase: reset!, reward, state, action_space, is_terminated
using Flux
import StatsBase: sample

const TOTAL_TIMESTEPS = 50_000
const BUFFER_SIZE = 10_000
const MIN_BUFF_SIZE = 200
const TRAIN_FREQ = 10
const TARGET_NET_FREQ = 100
const BATCH_SIZE = 128
const γ = 0.99

struct Transition{S}
  state::S
  action::Integer
  reward::AbstractFloat
  next_state::S
  terminal::Bool
end

struct TransitionBatch{S}  # Trajectory?
  states::Matrix{S}
  actions::Vector{<:Integer}
  rewards::Vector{<:AbstractFloat}
  next_states::Matrix{S}
  terminals::Vector{Bool}
end

struct ReplayBuffer
  capacity::Int
  data::AbstractVector{Transition}

  function ReplayBuffer(capacity)
    new(capacity, Vector{Transition}())
  end
end

function add!(rb::ReplayBuffer, data::Transition)
  if length(rb.data) > rb.capacity
    pop!(rb.data)
  end
  prepend!(rb.data, [data])
end

function sample(rb::ReplayBuffer, batch_size::Int)
  data = sample(rb.data, batch_size, replace=false)
  # TODO: should this be done in one loop?
  #  how does RL.jl do it?

  #  should ReplayBuffer just hold separate arrays for each item? Or a hashtable of arrays
  #  so you can add!(:action, action)
  states = map(t -> t.state, data)
  next_states = map(t -> t.next_state, data)
  rewards = map(t -> t.reward, data)
  actions = map(t -> t.action, data)
  terminals = map(t -> t.terminal, data)

  states = reduce(hcat, states)
  next_states = reduce(hcat, next_states)

  TransitionBatch(states, actions, rewards, next_states, terminals)
end

function linear_schedule(start_ϵ, end_ϵ, duration, t)
  slope = (end_ϵ - start_ϵ) / duration
  max(slope * t + start_ϵ, end_ϵ)
end


function dqn()
  env = CartPoleEnv()
  q_net = Chain(Dense(4, 120, relu), Dense(120, 84, relu), Dense(84, 2))
  opt = ADAM()
  target_net = deepcopy(q_net)

  rb = ReplayBuffer(BUFFER_SIZE)

  ϵ_schedule = t -> linear_schedule(1.0, 0.05, 10_000, t)

  ep_rew = 0

  reset!(env)
  for global_step in 1:TOTAL_TIMESTEPS
    obs = deepcopy(state(env))  # state needs to be coppied otherwise state and next_state is the same

    ϵ = ϵ_schedule(global_step)
    action = if rand() < ϵ
      env |> action_space |> rand
    else
      qs = q_net(obs)
      argmax(qs)
    end

    env(action)

    sart = Transition(
      obs,
      action,
      reward(env),
      deepcopy(state(env)),
      is_terminated(env)
    )
    add!(rb, sart)

    ep_rew += reward(env)
    if is_terminated(env)
      @show ep_rew
      ep_rew = 0
      reset!(env)
    end


    if (global_step > MIN_BUFF_SIZE) && (global_step % TRAIN_FREQ == 0)
      data = sample(rb, BATCH_SIZE)
      actions = CartesianIndex.(data.actions, 1:length(data.actions))

      next_q = data.next_states |> target_net |> eachcol .|> maximum
      td_target = data.rewards + γ * next_q .* (1.0 .- data.terminals)

      params = Flux.params(q_net)
      gs = gradient(params) do
        q = data.states |> q_net
        q = q[actions]
        Flux.mse(td_target, q)
      end

      Flux.Optimise.update!(opt, params, gs)

      if global_step % TARGET_NET_FREQ == 0
        target_net = deepcopy(q_net)
      end
    end
  end
end

dqn()
