module Buffer

import StatsBase: sample
using EllipsisNotation

"""Returns the size of x, if x is a scalar it's size is (1,)."""
_size_in_buffer(x::Union{AbstractArray,Real}) = ndims(x) == 0 ? (1,) : size(x)

mutable struct ReplayBuffer{TupleNames,TupleValues}
  data::NamedTuple{TupleNames,TupleValues}  # replay data
  capacity::Int  # max buffer size
  ptr::Int  # pointer to current index
  size::Int  # current array size

  function ReplayBuffer(transition::NamedTuple, capacity::Int)
    data = map(x -> zeros(eltype(x), (_size_in_buffer(x)..., capacity)), transition)
    new{keys(data),typeof(values(data))}(data, capacity, 1, 0)
  end
end



function add!(rb::ReplayBuffer{TupleNames,A}, transition::NamedTuple{TupleNames,B}) where {TupleNames} where {A} where {B}
  @inbounds for (key, value) in pairs(transition)
    # it would be nice to add an inbounds here, but I often 
    # have size mismatches when debugging and that will get in the way
    # @show key, size(value), size(rb.data[key])
    rb.data[key][.., rb.ptr] = value
  end

  # increment/wrap `ptr`
  rb.ptr = rb.ptr + 1 > rb.capacity ? 1 : rb.ptr + 1
  # increment `size` up to a max of `capacity`
  rb.size = min(rb.capacity, rb.size + 1)

  rb
end

# How to remove items for policy grad methods
function sample(rb::ReplayBuffer, n::Int; ordered=false)
  @assert n <= rb.size

  inds = sample(1:rb.size, n, replace=false, ordered=ordered)
  @inbounds map(x -> x[.., inds], rb.data)
end

function Base.last(rb::ReplayBuffer)
  @assert rb.size > 0
  i = rb.ptr - 1 < 1 ? rb.capacity : rb.ptr - 1
  @inbounds map(x -> x[.., i], rb.data)
end

function clear!(rb::ReplayBuffer)
  @inbounds for key in keys(rb.data)
    rb.data[key][:] = similar(rb.data[key])
  end

  rb.size = 0
  rb.ptr = 1
end

end # module

