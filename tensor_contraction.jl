using ITensors
using CSV, DataFrames

function transfer_matrix(DU::AbstractArray{T}, x::T; horizontal::Bool = true) where T <: Integer
    ###
end

function bricklayer(DU::AbstractArray{T}, σ::AbstractArray{T}, x::Integer) where T <: Integer
    i = Index(4,"index_i")
    j = Index(4, "index_j")

    A = ITensor(σ,i)
    U = ITensor(DU,i,j)

    sites = siteinds(2,x)

    cutoff = 1E-8
    maxdim = 10
    M = MPS(U,sites;cutoff=cutoff,maxdim=maxdim)

    return M
end

function test(h::Integer,v::Integer)
    W = CSV.read("data/FoldedTensors/DU_2_3806443768.csv",DataFrame)
    W = replace.(W, "(" => "", ")" => "")
    W = parse.([Complex{Float64}],W)
    Matrix(W)

    for i in 1:h
        for j in 1:v
            u = Index(q,"index_" * string(2 * i + 1) * "," * string(2 * j))
            v = Index(q,"index_" * string(2 * i) * "," * string(2 * j + 1))
            w = Index(q,"index_" * string(2 * i + 2) * "," * string(2 * j + 1))
            l = Index(q,"index_" * string(2 * i + 1) * "," * string(2 * j))


end


# DU = [1.0 0.0 0.0 0.0 ;0.0 1.0 0.0 0.0 ;0.0 0.0 1.0 0.0 ;0.0 0.0 0.0 1.0]
# σ = [1.0 0.0 0.0 -1.0]

# @show bricklayer(DU,σ,4)

@show test()