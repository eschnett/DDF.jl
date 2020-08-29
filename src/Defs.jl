module Defs

using StaticArrays

export invariant
function invariant end

if VERSION < v"1.5"
    Base.:≈(x) = y -> y ≈ x
end

end
