#load "..\Scripts\SetEnv.fsx"
open FsCNTK
open FsCNTK.FsBase
open Layers_Dense
open Layers_Sequence
open Models_Attention
open CNTK
open System.IO
open System
open Blocks


let flag = V.toValue [10.; -1.; -0.; 0.3; 100.]
let value_if_true = V.toValue [1.; 10.; 100.; 1000.; 10000.]
let value_if_false = V.toValue [ 2.; 20.; 200.; 2000.; 20000.]
let v = O.element_select(Node.Const flag.Data, Node.Const value_if_true.Data, Node.Const value_if_false.Data)
let v2 =  v |> E.eval1(idict[])