module ConfigParser
"""Allows for parsing a struct defined using the Base.@kwef macro into an command line args"""
# not entirely convinced this is the best option, it keeps the code nice and concise,
#  but doesn't allow for help text in CLI args

using ArgParse

export argparse_struct

function _dict_to_struct(params::Dict, struct_type::Type)
  struct_type(; params...)
end

function _struct_to_dict(s)
  return Dict(key => getfield(s, key) for key in propertynames(s))
end

function argparse_struct(s)
  """Gets the fields from a struct defined with Base.@kwdef and converts them to
  command line options which can be overriden and are then returned in the same struct
  """
  StructType = typeof(s)
  settings = ArgParseSettings()

  # create list for add_arg_table to parse
  args = []
  for attrib_name in propertynames(s)
    push!(args, "--$(string(attrib_name))")
    attrib_value = getfield(s, attrib_name)
    attrib_type = fieldtype(StructType, attrib_name)
    push!(args, Dict(:default => attrib_value, :arg_type => attrib_type))
  end

  add_arg_table(settings, args...)
  args = parse_args(settings)
  # converting string keys to symbols for dict_to_struct
  args = Dict(Symbol(k) => v for (k, v) in args)

  _dict_to_struct(args, StructType)
end

end  # module
