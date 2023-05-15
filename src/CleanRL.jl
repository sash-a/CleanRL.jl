module CleanRL

using ReinforcementLearningEnvironments: CartPoleEnv
using ReinforcementLearningBase: reset!, reward, state, is_terminated, action_space, state_space, AbstractEnv

using Flux
using StatsBase: sample, Weights, loglikelihood, mean, entropy, std
using Random: shuffle
using Distributions: Categorical

using ArgParse
using Dates: now, format

using LoggingExtras: TeeLogger, FormatLogger, ConsoleLogger
using LoggingFormats: JSON
using TensorBoardLogger: TBLogger

include("utils/replay_buffer.jl")
include("utils/config_parser.jl")
include("utils/logger.jl")
include("utils/networks.jl")

include("algorithms/ppo2.jl")
include("algorithms/dqn.jl")
include("algorithms/a2c.jl")
end # module
