module Logger

using LoggingExtras: TeeLogger, FormatLogger, ConsoleLogger
using LoggingFormats: JSON
using TensorBoardLogger: TBLogger

export make_logger

function make_logger(run_name; to_terminal=true, to_tensorboard=true, to_json=false)
  """Creates a logger and sets it as the global_logger"""
  json_log_file = "logs/dqn|$(run_name).json"
  tb_log_file = "logs/dqn|$(run_name)"

  loggers = []

  if to_terminal
    push!(loggers, ConsoleLogger())
  end
  if to_tensorboard
    push!(loggers, TBLogger(tb_log_file))
  end
  if to_json
    push!(loggers, FormatLogger(JSON(), json_log_file; append=true))
  end

  logger = TeeLogger(loggers...)
  Base.global_logger(logger)  # set this logger to be the global logger

  logger
end
end  # module
