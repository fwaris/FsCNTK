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
    | LeakyReLU
    | PReLU of float

//based on python layers module (see CNTK Python API for documentation)
//mimics python code as closely as feasible

module Layers =
  let trace = ref false

  let inline asList sz x  = [for _ in 1 .. sz -> x]

  let inline idict (s:(^a * ^b) seq) =
      let d = new System.Collections.Generic.Dictionary< ^a, ^b>()
      s |> Seq.iter d.Add
      d

  let internal addActivation (v:Variable) = function
      | Activation.NONE       ->              !>v
      | Activation.ReLU       -> C.ReLU       v
      | Activation.LeakyReLU  -> C.LeakyReLU  v
      | Activation.Sigmoid    -> C.Sigmoid    v
      | Activation.Tanh       -> C.Tanh       v
      | Activation.PReLU c    -> let alpha = new Constant(v.Shape, dataType, c)
                                 C.PReLU(alpha,v)

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
      Layers.addActivation n.Var actType |> F
        
