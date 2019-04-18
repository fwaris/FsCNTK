#load "..\Scripts\SetEnv.fsx"
open FsCNTK

let flag = [10.; -1.; -0.; 0.3; 100.]
let value_if_true =  [1.; 10.; 100.; 1000.; 10000.]
let value_if_false = [ 2.; 20.; 200.; 2000.; 20000.]
let v = O.element_select(Node.Const flag, Node.Const value_if_true, Node.Const value_if_false)
let v2 =  v |> E.eval1(idict[])