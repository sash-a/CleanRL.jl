module Networks

using ReinforcementLearningBase: action_space, state_space, AbstractEnv
using Flux

function make_actor_critic_nn(env::AbstractEnv)
  # this doesn't seem to be a general way to get in and out size for all envs
  in_size = length(state_space(env).s)
  out_size = length(action_space(env).s)
  ob_net = Chain(Dense(in_size, 120, relu), Dense(120, 84, relu))
  actor = Chain(ob_net, Dense(84, out_size), softmax)
  critic = Chain(ob_net, Dense(84, 1))

  actor, critic
end

end  # module
