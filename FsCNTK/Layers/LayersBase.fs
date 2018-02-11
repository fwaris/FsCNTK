namespace FsCNTK
open CNTK
open System
open FsBase
open Blocks

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

module Layers =
  let trace = ref false

  let inline asList sz x  = [for _ in 1 .. sz -> x]

  let inline idict (s:(^a * ^b) seq) =
      let d = new System.Collections.Generic.Dictionary< ^a, ^b>()
      s |> Seq.iter d.Add
      d

  let internal addActivation (n:Node) = function
      | Activation.NONE        ->              n
      | Activation.ReLU        -> C.ReLU       n.Var   |> F
      | Activation.LeakyReLU c -> C.LeakyReLU(n.Var,float32 (match c with None->0.3 | Some c ->c))   |> F
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
      if !Layers.trace then printfn ">> Activation"
      Layers.addActivation n actType
        
