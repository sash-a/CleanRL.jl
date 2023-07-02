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

function make_actor_critic_shared(action_space, obs_space, hidden_sizes::Vector{Int}=Int[64, 64])
  in_size = length(obs_space)
  out_size = length(action_space)

  ob_net = mlp(vcat(in_size, hidden_sizes))

  actor_final_gain = Flux.orthogonal(; gain=0.01)
  critic_final_gain = Flux.orthogonal(; gain=1.0)
  actor_final_layer = Dense(last(hidden_sizes), out_size; init=actor_final_gain)
  critic_final_layer = Dense(last(hidden_sizes), 1; init=critic_final_gain)

  actor = Chain(ob_net, actor_final_layer, softmax)
  critic = Chain(ob_net, critic_final_layer)

  actor, critic
end

function make_actor_critic_shared(env::AbstractEnv, hidden_sizes::Vector{Int}=Int[64, 64])
  make_actor_critic_shared(action_space(env), state_space(env), hidden_sizes)
end

function make_actor_critic(action_space, obs_space, hidden_sizes::Vector{Int}=Int[64, 64])
  in_size = length(obs_space)
  out_size = length(action_space)

  actor_final_gain = Flux.orthogonal(; gain=0.01)
  critic_final_gain = Flux.orthogonal(; gain=1.0)
  actor_final_layer = Dense(last(hidden_sizes), out_size; init=actor_final_gain)
  critic_final_layer = Dense(last(hidden_sizes), 1; init=critic_final_gain)

  actor = Chain(mlp(vcat(in_size, hidden_sizes)), actor_final_layer, softmax)
  critic = Chain(mlp(vcat(in_size, hidden_sizes)), critic_final_layer)

  actor, critic
end

function make_actor_critic(env::AbstractEnv, hidden_sizes::Vector{Int}=Int[64, 64])
  make_actor_critic(action_space(env), state_space(env), hidden_sizes)
end

end  # module
