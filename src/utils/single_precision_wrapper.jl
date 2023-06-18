# possibly not needed a lot of environments use either Env{T=Float32}(...) or Env(T=Float32, ...)
# so can specify like that instead of using this wrapper, however this wrapper makes things more
# consistent 
struct SinglePrecisionEnv{E<:AbstractEnv} <: AbstractEnvWrapper
  env::E
  precision::Type{<:AbstractFloat}

  function SinglePrecisionEnv(env::E; precision::Type{<:AbstractFloat}=Float32) where {E<:AbstractEnv}
    new{E}(env, precision)
  end
end

# todo: how to change precision of state space?

RLBase.state(env::SinglePrecisionEnv, args...; kwargs...) =
  env.precision.(state(env.env, args...; kwargs...))

RLBase.reward(env::SinglePrecisionEnv, args...; kwargs...) =
  env.precision.(reward(env.env, args...; kwargs...))
