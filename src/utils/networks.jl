module Networks

using ReinforcementLearning
using Flux

function mlp(layer_sizes::Vector{Int})
  layers = []
  for i in 1:length(layer_sizes)-1
    push!(layers, Dense(layer_sizes[i], layer_sizes[i+1], relu))
  end
  Chain(layers...)
end

function make_actor_critic_shared(action_space, obs_space, hidden_sizes::Vector{Int}=Int[64, 64])
  in_size = length(obs_space)
  out_size = length(action_space)

  ob_net = mlp(vcat(in_size, hidden_sizes))

  actor = Chain(ob_net, Dense(last(hidden_sizes), out_size), softmax)
  critic = Chain(ob_net, Dense(last(hidden_sizes), 1))

  actor, critic
end

function make_actor_critic_shared(env::AbstractEnv, hidden_sizes::Vector{Int}=Int[64, 64])
  make_actor_critic_shared(action_space(env), state_space(env), hidden_sizes)
end

function make_actor_critic(action_space, obs_space, hidden_sizes::Vector{Int}=Int[64, 64])
  in_size = length(obs_space)
  out_size = length(action_space)

  actor = Chain(mlp(vcat(in_size, hidden_sizes)), Dense(last(hidden_sizes), out_size), softmax)
  critic = Chain(mlp(vcat(in_size, hidden_sizes)), Dense(last(hidden_sizes), 1))

  actor, critic
end

function make_actor_critic(env::AbstractEnv, hidden_sizes::Vector{Int}=Int[64, 64])
  make_actor_critic(action_space(env), state_space(env), hidden_sizes)
end

end  # module
