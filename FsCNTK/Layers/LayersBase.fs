﻿namespace FsCNTK
open CNTK
open System
open FsBase
open Blocks
open FsBase

  //layers type
type Activation = 
    | NONE
    | ReLU
    | Sigmoid
    | Tanh
    | LeakyReLU of float option
    | PReLU of float
    | SELU
    | ELU

//based on python layers module (see CNTK Python API for documentation)
//mimics python code as closely as feasible

[<AutoOpen>]
module Layers =
  let trace = ref false

  let internal addActivation (n:Node) = function
      | Activation.NONE        ->              n
      | Activation.ReLU        -> C.ReLU       n.Var   |> F
      | Activation.LeakyReLU c -> C.LeakyReLU(n.Var, (*float32*) (match c with None->  0.3 | Some c ->  c))   |> F
      | Activation.Sigmoid     -> C.Sigmoid    n.Var   |> F
      | Activation.Tanh        -> C.Tanh       n.Var   |> F
      | Activation.SELU        -> C.SELU       n.Var   |> F
      | Activation.ELU         -> C.ELU        n.Var   |> F
      | Activation.PReLU c     -> let alpha = new Constant(!--(O.shape n), dataType, c)
                                  C.PReLU(alpha,n.Var) |> F

type L =

  static member _window (x:Variable, axis, _begin, _end, step, stride, ?initial_state) = 
      if stride <> 1 then failwith "windowed convolution with stride not yet implemented"
      let initial_state = initial_state |> Option.defaultValue null
      let shifted =
          [|
              for t in _begin .. step .. _end do
                  yield 
                      match t with
                      | 0             -> x
                      | t when t < 0 -> !> C.PastValue(x,initial_state, uint32 -t)
                      | t            -> !> C.FutureValue(x,initial_state,uint32 t)
          |]
      C.Splice(varVector shifted, axis)

  static member Activation actType (n:Node) =
      if !Layers.trace then printfn ">> Activation %A" actType
      Layers.addActivation n actType
        
  static member Embedding
    (
      ?shape,
      ?init,
      ?weights:NDArrayView,
      ?name
    )
    =
    match init,weights with Some _, Some _ -> failwith "Embedding: init and weights options are mutually exclusive" | _ -> ()
    let name = defaultArg name "E"
    let init = defaultArg init (C.GlorotUniformInitializer())

    let E = 
        match weights, shape with 
        | None, None      -> failwith "Embedding: output shape must be specified if weights are not given" 
        | Some w, Some _  -> failwith "Embedding: output shape must not be specified when weights are given"
        | Some w, None    -> new Constant(w,name) :> Variable |> V
        | None, Some shp  ->  Node.Parm( (D NDShape.InferredDimension) + shp, init, name)

    fun (x:Node) -> 
      let r = x * E
      if !Layers.trace then printfn ">> Embedding %A" (O.shape r)
      r
      
  static member Label name =
    fun (x:Node) -> 
      if !Layers.trace then printfn ">> Label %A" (O.shape x)
      C.Alias(x.Var,name) |> F