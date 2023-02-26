module Buffer2

import StatsBase: sample

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
  inds = sample(1:rb.size, n, replace=false, ordered=ordered)
  map(x -> x[inds, :], rb.data)
end

end # module

# b = Buffer.Buff((action=1, obs=[1, 2, 3]), 3)
# Buffer.add!(b, (action=2, obs=[4, 5, 6]))
# Buffer.add!(b, (action=3, obs=[7, 8, 9]))
# Buffer.add!(b, (action=4, obs=[10, 11, 12]))
# Buffer.add!(b, (action=5, obs=[13, 14, 15]))
# @show b
#
# s = Buffer.sample(b, 2)
# @show s size(s.action) size(s.obs)
