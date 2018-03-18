#load "..\Scripts\SetEnv.fsx"
open FsCNTK
open FsCNTK.FsBase
open FsCNTK.Layers
open Layers_Dense
open Layers_Dropout
open CNTK
open System.IO
open FsCNTK.FsBase
open Layers_Recurrence
open System.Collections.Generic
open Probability
open System
open FsCNTK.Layers_Recurrence

type CNTKLib = C

let m1 = Function.Load(@"D:\repodata\fscntk\l_fs_m.bin",device)
let m2 = Function.Load(@"D:\repodata\fscntk\l_py_m.bin",device)


let v1 = Node.Input(D 1, name="v1")
let v2 = Node.Input(D 1, name="v2")
let v3 = Node.Input(D 1, name="v3")
let p = v1 + v2
let c = O.combine [p;v3]
let f = c.Func
f.Outputs.Count
let n1 = O.getOutput 0 c
let n2 = O.getOutput 1 c
n1.Var
n2.Var
v3.Var
m1.Inputs |> Seq.map (fun i-> !++ i.Shape, i.Name, i.Uid) |> Seq.toArray |> Seq.iter (printfn "%A")
m2.Inputs |> Seq.map (fun i-> !++ i.Shape, i.Name, i.Uid) |> Seq.toArray |> Seq.iter (printfn "%A")
