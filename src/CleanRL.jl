module CleanRL

# must be before using ReinforcementLearning for gym envs
using PyCall
using ReinforcementLearning

using Flux
using Flux: Zygote
using StatsBase: sample, Weights, loglikelihood, mean, entropy, std
using Random: shuffle
using Distributions: Categorical

using Dates: now, format

include("utils/replay_buffer.jl")
include("utils/config_parser.jl")
include("utils/logger.jl")
include("utils/networks.jl")

include("algorithms/dqn.jl")
include("algorithms/ddpg.jl")
include("algorithms/a2c.jl")
include("algorithms/ppo.jl")

end # module
