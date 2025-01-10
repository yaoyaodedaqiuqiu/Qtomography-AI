# function CorrFuncZZ(L::Number)

#     Tree = ObservableTree()
#     for i in 1:L-1
#          addObs!(Tree, (Sz, Sz), (1, i+1), (false, false); name = (:Sz, :Sz))
#     end

#     return Tree

# end

function extract_prob(Obs)

     v = 0.0
     for outer_key in keys(Obs)
         inner_dict = Obs[outer_key]
         for inner_key in keys(inner_dict)
             v = inner_dict[inner_key]
             # println("Inner Key: ", inner_key, ", Value: ", inner_dict[inner_key])
         end
     end
 
     return v
 end
 