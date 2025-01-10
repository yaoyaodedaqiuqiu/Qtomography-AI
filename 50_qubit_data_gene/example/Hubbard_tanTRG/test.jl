using MKL
using FiniteMPS, FiniteLattices
using Distributions
using JSON

# define Pauli Z and X
using TensorKit
const pspace = ℂ^2
const Sz = let
     mat = Float64[1 0; 0 -1]
     TensorMap(mat, pspace, pspace)
end
const Sx = let
     mat = Float64[0 1; 1 0]
     TensorMap(mat, pspace, pspace)
end
const P0 = let
     mat = Float64[1 0; 0 0]
     TensorMap(mat, pspace, pspace)
end
const P1 = let
     mat = Float64[0 0; 0 1]
     TensorMap(mat, pspace, pspace)
end

include("CorrFunc.jl")
include("isingmodel.jl")

verbose = 0
GCstep = true
# lsβ = vcat(2.0 .^ (-15:2:-1), 1:8)
lsβ = vcat((0:0.05:0.45))
# lsβ = vcat((0:0.25:0.25))  # only for test mode
lsF = zeros(length(lsβ))
lsE = zeros(length(lsβ))
lsD = map(lsβ) do β
     β < 1 && return 500
     D = ceil(β/2; digits = 0) * 1000 |> Int
     min(D, 4000)
end
L = 4    # number of qubits
J = 1.0
h = 0.0
trunc_dim = 256

# Prepare Hamiltonian
H = AutomataMPO(NNIsing(L,J,h))  # make MPO for H
println(Root)

ρ = let
     # ============= SETTN ===============
     ρ, lsF_SETTN = SETTN(H, lsβ[1];
          maxorder=2, verbose=verbose,
          GCstep=false, GCsweep=true, tol=1e-8,
          compress = 1e-12,
          trunc=truncdim(trunc_dim),
          maxiter = 4)
     lsF[1] = lsF_SETTN[end]
     lnZ = 2 * log(norm(ρ))
     normalize!(ρ)
     # ============= TDVP ==============
     Env = Environment(ρ', H, ρ)
     canonicalize!(Env, 1)
     lsE[1] = scalar!(Env; normalize=true)
     let 
          println("β = $(lsβ[1]), F = $(lsF[1]), E = $(lsE[1])")
          flush(stdout)
     end

     for i = 2:length(lsβ)
          D = lsD[i]
          dβ = lsβ[i] - lsβ[i-1]
          TDVPSweep1!(Env, dβ;
               CBEAlg = CheapCBE(D + div(D, 4), 1e-8),
               GCstep=GCstep, GCsweep=true, verbose=verbose,
               trunc=truncdim(D) & truncbelow(1e-12))
          lnZ += 2 * log(norm(ρ))
          normalize!(ρ)
          lsF[i] = -lnZ / lsβ[i]
          lsE[i] = scalar!(Env; normalize=true)
          let 
               println("β = $(lsβ[i]), F = $(lsF[i]), E = $(lsE[i])")
               flush(stdout)
          end
     end
     ρ
end

################# Z_1 Z_j ##########################
# Calculate Obs Z_1 Z_j
Tree = ObservableTree()
for i in 1:L-1
     addObs!(Tree, (Sz, Sz), (1, i+1), (false, false); name = (:Sz, :Sz))
end
calObs!(Tree, ρ)
Obs = convert(NamedTuple, Tree)
println(Obs)


################# Sampling via diagonal ############
total_bitstring = []
for a = 1:1000
     bitstring = [] # ininialize the sampled bitstring
     for ind = 1:L
          println("round:$ind, bitsting = $bitstring")
          if ind == 1
               TreeSample = ObservableTree()
               addObs!(TreeSample, (P0), (1); name = (:P0))
               calObs!(TreeSample, ρ) # calculate the observables
               sample_obs = convert(NamedTuple, TreeSample) # Convert the results into a usable format
               println(sample_obs)

               # extract probabilities
               prob_P0 = extract_prob(sample_obs) # probability for current P0
               prob_P1 = 1 - prob_P0 # probability for current P1
          else
               previous_operators = [bit == 0 ? P0 : P1 for bit in bitstring]  # Previous bits' operators
               previous_name_tuple = [Symbol(bit == 0 ? "P0" : "P1") for bit in bitstring]
               Findix = Tuple([false for i in 1:ind])  # Each index in its own tuple, e.g., (1,), (2,), ...
               indices = Tuple([i for i in 1:ind])  # Each index in its own tuple, e.g., (1,), (2,), ...

               current_oprP0 = P0
               current_oprP1 = P1

               all_oprP0 = vcat(previous_operators, [current_oprP0])  # Combine previous and current
               all_oprP1 = vcat(previous_operators, [current_oprP1])  # Combine previous and current
               
               # Construct the name as a tuple of Strings
               current_symbolP0 = Symbol("P0")
               current_symbolP1 = Symbol("P1")

               name_tupleP0 = vcat(previous_name_tuple, current_symbolP0)
               name_tupleP1 = vcat(previous_name_tuple, current_symbolP1)

               # println(all_oprP0)
               # println(all_oprP1)

               TreeSampleP0 = ObservableTree()
               TreeSampleP1 = ObservableTree()

               addObs!(TreeSampleP0, Tuple(all_oprP0), indices, Findix; name=Tuple(name_tupleP0))
               addObs!(TreeSampleP1, Tuple(all_oprP1), indices, Findix; name=Tuple(name_tupleP1))
               calObs!(TreeSampleP0, ρ)
               calObs!(TreeSampleP1, ρ)
               
               sample_obsP0 = convert(NamedTuple, TreeSampleP0)
               sample_obsP1 = convert(NamedTuple, TreeSampleP1)

               # println(sample_obsP0)
               # println(sample_obsP0)

               prob_P0 = extract_prob(sample_obsP0)
               prob_P1 = extract_prob(sample_obsP1)
          end

          # Print probabilities for debugging
          println("Qubit $ind: P0 = $prob_P0, P1 = $prob_P1")
          
          # Create a categorical distribution and sample
          norm_factor = prob_P0 + prob_P1 # prevent prob != 1
          dist = Categorical([prob_P0/norm_factor, prob_P1/norm_factor])
          current_bit = rand(dist) - 1  # Categorical returns 1 or 2; adjust to 0 or 1
          println("Selected Qubit $current_bit")
          push!(bitstring, current_bit)  # Append the result to the bitstring
     end

     println("Final round: bitsting = $bitstring")
     push!(total_bitstring, bitstring)
end
println("total_bitstring $total_bitstring")

json_file = "total_bitstring.json"
open(json_file, "w") do io
    JSON.print(io, total_bitstring)
end