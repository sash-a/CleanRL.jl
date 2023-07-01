module CleanRL

using Base: Threads
using Dates: now, format
using Random: shuffle, Xoshiro

# must be before using ReinforcementLearning for gym envs
# using PyCall
using ReinforcementLearning
# using GridWorlds
#
using Flux
using Flux: Zygote
using StatsBase: sample, Weights, loglikelihood, mean, entropy, std
using Random: shuffle
using Distributions: Categorical, logpdf

include("utils/replay_buffer.jl")
include("utils/config_parser.jl")
include("utils/logger.jl")
include("utils/networks.jl")
include("utils/multi_thread_env.jl")
include("utils/single_precision_wrapper.jl")

include("algorithms/dqn.jl")
include("algorithms/ddpg.jl")
include("algorithms/a2c.jl")
include("algorithms/ppo.jl")

end # module
