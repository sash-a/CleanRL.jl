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

function make_actor_critic_shared(env::AbstractEnv, hidden_sizes::Vector{Int}=Int[64, 64])
  in_size = length(state_space(env))
  out_size = length(action_space(env))

  ob_net = mlp(vcat(in_size, hidden_sizes))

  actor = Chain(ob_net, Dense(last(hidden_sizes), out_size), softmax)
  critic = Chain(ob_net, Dense(last(hidden_sizes), 1))

  actor, critic
end

function make_actor_critic(env::AbstractEnv, hidden_sizes::Vector{Int}=Int[64, 64])
  in_size = length(state_space(env))
  out_size = length(action_space(env))

  actor = Chain(mlp(vcat(in_size, hidden_sizes)), Dense(last(hidden_sizes), out_size), softmax)
  critic = Chain(mlp(vcat(in_size, hidden_sizes)), Dense(last(hidden_sizes), 1))

  actor, critic
end

end  # module
