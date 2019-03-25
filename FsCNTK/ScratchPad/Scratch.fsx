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

let x = Node.Input(D 512, dynamicAxes=[Axis.DefaultDynamicAxis(); Axis.DefaultBatchAxis()])
let ax = -2
let ax' = -ax - 1
let ab = L.PastValueWindow(20, new Axis(ax), false) (x)
//a.DebugDisplay
//b.DebugDisplay