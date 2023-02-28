module Buffer

import StatsBase: sample
using InvertedIndices

mutable struct ReplayBuffer
  data::NamedTuple  # replay data
  capacity::Int  # max buffer size
  ptr::Int  # pointer to current index
  size::Int  # current array size

  function ReplayBuffer(transition::NamedTuple, capacity::Int)
    data = map(x -> zeros(eltype(x), (capacity, size(x)...)), transition)
    new(data, capacity, 1, 0)
  end
end

function add!(rb::ReplayBuffer, transition::NamedTuple)
  @assert keys(rb.data) == keys(transition)

  for (key, value) in pairs(transition)
    rb.data[key][rb.ptr, :] .= value
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
  map(x -> x[inds, :], rb.data)
end

function sample_and_remove(rb::ReplayBuffer, n::Int; ordered=false)
  @assert n <= rb.size

  inds = sample(1:rb.size, n, replace=false, ordered=ordered)
  data = map(x -> x[inds, :], rb.data)

  # remove data
  for key in keys(rb.data)
    # Two coppies here, quite inefficient, but NamedTuples don't let you set fields
    keep = rb.data[key][Not(inds), :]
    rb.data[key] .= zeros(size(rb.data[key]))
    rb.data[key][1:size(keep)[1], :] .= keep
  end

  rb.size -= n
  rb.ptr -= n

  data
end

end # module

