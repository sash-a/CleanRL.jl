module Networks

using ReinforcementLearning
using Flux

function mlp(layer_sizes::Vector{Int}, activation=tanh_fast, gain=sqrt(2))
  layers = []
  for i in 1:length(layer_sizes)-1
    init = Flux.orthogonal(; gain=gain)
    push!(layers, Dense(layer_sizes[i], layer_sizes[i+1], activation; init=init))
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

  actor_final_gain = Flux.orthogonal(; gain=0.01)
  critic_final_gain = Flux.orthogonal(; gain=1.0)
  actor_final_layer = Dense(last(hidden_sizes), out_size; init=actor_final_gain)
  critic_final_layer = Dense(last(hidden_sizes), 1; init=critic_final_gain)

  actor = Chain(mlp(vcat(in_size, hidden_sizes)), actor_final_layer)
  critic = Chain(mlp(vcat(in_size, hidden_sizes)), critic_final_layer)

  actor, critic
end

end  # module
