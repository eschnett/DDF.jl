struct Domain{D, T}
    xmin::NTuple{D, T}
    xmax::NTuple{D, T}
end



# TODO: Coordinates are functions on the manifold; represent them as
# Fun
struct Coords{D, T}
    mf::Manifold{D}
    dom::Domain{D}
    coords::Vector{NTuple{D, T}}
end



@computed struct Tree{D, T}
    dom::Domain{D}
    mf::Manifold{D}

    pivot::NTuple{D, T}
    tree::Union{NTuple{2^D, fulltype(Tree{D, T})},
                Vector{Int}}
end



# function sample
# function evaluate
