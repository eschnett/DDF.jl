struct Domain{D, T}
    xmin::NTuple{D, T}
    xmax::NTuple{D, T}
end



struct Coords{D, T}
    mf::Manifold{D}
    dom::Domain{D}
    coords::Vector{NTuple{D, T}}
end



struct Tree{D, T}
    dom::Domain{D}
    mf::Manifold{D}

    pivot::NTuple{D, T}
    tree::Union{Vector{Tree{D, T}}, # 2^D elements
                Vector{Int}}
end



# function sample
# function evaluate
