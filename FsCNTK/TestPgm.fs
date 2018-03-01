module Pgm
//#load "..\Scripts\SetEnv.fsx"
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
let m1 = Function.Load(@"D:\repodata\fscntk\l_fs_m.bin",device)
let m2 = Function.Load(@"D:\repodata\fscntk\l_py_m.bin",device)
let mb = m2.RootFunction.Inputs.[2].Owner
let mbo = mb.BlockRoot()
mbo.Save(@"D:\repodata\fscntk\mbo_m.bin")
let _ = Function.Load(@"D:\repodata\fscntk\mbo_m.bin",device)
(*
- function has arguments and inputs
- inputs are nodes from outside the function
- for graph root, input are prob. just parameters and constants
- for graph root, 'rootfunction' holds outpur var
  and points to other input nodes
- root function will point to itself for non root nodes?
- inputs are parameters, constants or output variables 
- output vars have owners which are functions
*)


let i = 1
()