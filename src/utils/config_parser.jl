module ConfigParser
"""ALlows for parsing a struct defined using the Base.@kwef macro into an command line args"""
# not entirely convinced this is the best option, it keeps the code nice and concise,
#  but doesn't allow for help text in CLI args

using ArgParse

export argparse_struct

function dict_to_struct(params::Dict, struct_type::Type)
  struct_type(; params...)
end

function struct_to_dict(s)
  return Dict(key => getfield(s, key) for key in propertynames(s))
end

function argparse_struct(s)
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

  dict_to_struct(args, StructType)
end

end  # module
