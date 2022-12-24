module Buffers

import StatsBase: sample
using DataStructures: CircularBuffer

export Transition, Trajectory, PGTransition, PGTrajectory, ReplayBuffer, ReplayQueue, add!, sample

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

struct PGTransition{S} <: AbstractTransition
  state::S
  log_prob::AbstractFloat
  reward::AbstractFloat
  terminals::Bool
end

struct PGTrajectory{S} <: AbstractTransition
  state::Matrix{S}
  log_probs::Matrix{<:AbstractFloat}
  reward::Vector{AbstractFloat}
  terminals::Vector{Bool}
end

struct ReplayBuffer{T} <: AbstractReplayBuffer where {T<:AbstractTransition}
  capacity::Int
  data::CircularBuffer{T}

  function ReplayBuffer{T}(capacity) where {T<:AbstractTransition}
    new{T}(capacity, CircularBuffer{T}(capacity))
  end
end

struct ReplayQueue{T} <: AbstractReplayBuffer where {T<:AbstractTransition}
  data::Vector{T}

  function ReplayQueue{T}() where {T<:AbstractTransition}
    new{T}(Vector{T}())  # how to make this generic?
  end
end

function add!(rb::AbstractReplayBuffer, data::AbstractTransition)
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

function sample(rb::ReplayQueue, batch_size::Int)
  data = rb.data[1:batch_size]
  rb.data = rb.data[batch_size+1:end]

  states = map(t -> t.state, data)
  log_probs = map(t -> t.log_prob, data)
  rewards = map(t -> t.reward, data)
  terminals = map(t -> t.terminal, data)

  states = reduce(hcat, states)
  log_probs = reduce(hcat, log_probs)

  return PGTrajectory(states, log_probs, rewards, terminals)
end

end # module
