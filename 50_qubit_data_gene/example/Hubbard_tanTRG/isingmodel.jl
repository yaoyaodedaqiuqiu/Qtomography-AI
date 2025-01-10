function NNIsing(L::Number, J::Number, h::Number)

     Root = InteractionTreeNode()
     for i in 1:L-1
          addIntr2!(Root, (Sz, Sz), (i, i+1), J; name = (:Sz, :Sz))
     end
     for i in 1:L
          addIntr1!(Root, (Sx), (i), h; name = (:Sx))
     end

     return InteractionTree(Root)

end
