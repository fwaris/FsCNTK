#load "..\Scripts\SetEnv.fsx"
open FsCNTK

let flag = [10.; -1.; -0.; 0.3; 100.]
let value_if_true =  [1.; 10.; 100.; 1000.; 10000.]
let value_if_false = [ 2.; 20.; 200.; 2000.; 20000.]
let v = O.element_select(Node.Const flag, Node.Const value_if_true, Node.Const value_if_false)
let v2 =  v |> E.eval1(idict[])

let c = Vl.toValue([0.;1.;4.;5.], Ds [2;1])
let x = Node.Input(Ds [2;1])
let y = Node.Const([for i in 0. .. 11. -> i]) |> O.reshapeF (Ds [6;2])
let m = O.gather(y,x) |> E.eval1 (idict[x.Var,c])
