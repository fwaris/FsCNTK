#load "..\Scripts\SetEnv.fsx"
open FsCNTK
open CNTK

//gather
let c = Vl.toValue([0.;1.;4.;5.], Ds [2;1])
let x = Node.Input(Ds [2;1])
let y = Node.Const([for i in 0. .. 11. -> i]) |> O.reshapeF (Ds [6;2])
let m = O.gather(y,x) 
let dt = m |> E.eval1 (idict[x.Var,c])

