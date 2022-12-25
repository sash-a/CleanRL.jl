module Logger

using LoggingExtras: TeeLogger, FormatLogger, ConsoleLogger
using LoggingFormats: JSON
using TensorBoardLogger: TBLogger

export make_logger

function make_logger(run_name; to_terminal=true, to_tensorboard=true, to_json=false)
  """Creates a logger and sets it as the global_logger"""
  loggers = []

  if to_terminal
    push!(loggers, ConsoleLogger())
  end
  if to_tensorboard
    push!(loggers, TBLogger("logs/$(run_name)"))
  end
  if to_json
    push!(loggers, FormatLogger(JSON(), "logs/$(run_name).json"; append=true))
  end

  logger = TeeLogger(loggers...)
  Base.global_logger(logger)  # set this logger to be the global logger

  logger
end
end  # module
