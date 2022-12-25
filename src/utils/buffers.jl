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

# Could combine these into 1 type if we made next states optional somehow
struct PGTransition{S} <: AbstractTransition
  state::S
  action::Integer
  reward::AbstractFloat
  terminal::Bool
end

struct PGTrajectory{S} <: AbstractTransition
  states::Matrix{S}
  actions::Vector{<:Integer}
  rewards::Vector{<:AbstractFloat}
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
  data = splice!(rb.data, 1:batch_size)

  states = map(t -> t.state, data)
  actions = map(t -> t.action, data)
  rewards = map(t -> t.reward, data)
  terminals = map(t -> t.terminal, data)

  states = reduce(hcat, states)

  return PGTrajectory(states, actions, rewards, terminals)
end

end # module
