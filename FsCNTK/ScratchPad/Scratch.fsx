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

type Tree = N of Node * Tree list



let rec buildTree m acc (f:Function) = 
  let m = m |> Map.add f.Uid (F f)
  if f.RootFunction <> null then
    buildTree



m1.Inputs |> Seq.map (fun i-> !++ i.Shape, i.Name, i.Uid) |> Seq.toArray |> Seq.iter (printfn "%A")
m2.Inputs |> Seq.map (fun i-> !++ i.Shape, i.Name, i.Uid) |> Seq.toArray |> Seq.iter (printfn "%A")
