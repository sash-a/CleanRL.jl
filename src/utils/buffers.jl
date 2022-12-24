module Buffers

import StatsBase: sample
using DataStructures: CircularBuffer

export Transition, Trajectory, ReplayBuffer, add!, sample

abstract type AbstractTransition end
abstract type AbstractTrajectory end
abstract type AbstractReplayBuffer end

# Results from a single step in RL env
struct Transition{S} <: AbstractTransition
  state::S
  action::Integer
  reward::AbstractFloat
  next_state::S
  terminal::Bool
end

# A batch of transitions
struct Trajectory{S} <: AbstractTrajectory
  states::Matrix{S}
  actions::Vector{<:Integer}
  rewards::Vector{<:AbstractFloat}
  next_states::Matrix{S}
  terminals::Vector{Bool}
end

struct ReplayBuffer <: AbstractReplayBuffer
  capacity::Int
  data::CircularBuffer{Transition}

  function ReplayBuffer(capacity)
    new(capacity, CircularBuffer{Transition}(capacity))
  end
end

function add!(rb::ReplayBuffer, data::AbstractTransition)
  """Add transisition to the replay buffer"""
  push!(rb.data, data)
end

function sample(rb::ReplayBuffer, batch_size::Int)
  """Get a random batch of transitions (Trajectory) from the replay buffer"""
  data = sample(rb.data, batch_size, replace=false)

  # TODO: should this be done in one loop?
  #  how does RL.jl do it?
  states = map(t -> t.state, data)
  next_states = map(t -> t.next_state, data)
  rewards = map(t -> t.reward, data)
  actions = map(t -> t.action, data)
  terminals = map(t -> t.terminal, data)

  states = reduce(hcat, states)
  next_states = reduce(hcat, next_states)

  Trajectory(states, actions, rewards, next_states, terminals)
end
end # module
