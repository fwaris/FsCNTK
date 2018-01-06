namespace FsCNTK
open CNTK
open System
open FsBase
open Blocks

//based on python layers module (see CNTK Python API for documentation)
//mimics python code closely

module Layers =
  let device =   DeviceDescriptor.UseDefaultDevice()
  let dataType = DataType.Float
  let inline asList sz x  = [for _ in 1 .. sz -> x]

  let inline idict (s:(^a * ^b) seq) =
      let d = new System.Collections.Generic.Dictionary< ^a, ^b>()
      s |> Seq.iter d.Add
      d

  let parmVector (ps:Parameter seq) = 
      let pv = new ParameterVector(Seq.length ps)
      ps |>  Seq.iter pv.Add
      pv

  let lrnVector (ls:Learner seq) =
      let lv = new LearnerVector(Seq.length ls)
      ls |>  Seq.iter lv.Add
      lv

  let boolVector (ls:bool seq) =
      let lv = new BoolVector(Seq.length ls)
      ls |>  Seq.iter lv.Add
      lv

  let prgwVector (pws:ProgressWriter seq) =
      let pwv = new ProgressWriterVector(Seq.length pws)
      pws |>  Seq.iter pwv.Add
      pwv

  let varVector (vs:Variable seq) =
      let vv = new VariableVector(Seq.length vs)
      vs |>  Seq.iter vv.Add
      vv

  let intVector (is:int seq) =
      let vs = new IntVector(Seq.length is)
      is |>  Seq.iter vs.Add
      vs

    //layers type
  type Activation = 
      | NONE
      | ReLU
      | Sigmoid
      | Tanh
      | LeakyReLU
      | PReLU of float

  type L =

    static member private _window (x:Variable, axis, _begin, _end, step, stride, ?initial_state) = 
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

    static member  activation v = function
      | Activation.NONE       ->              v
      | Activation.ReLU       -> C.ReLU       !>v
      | Activation.LeakyReLU  -> C.LeakyReLU  !>v
      | Activation.Sigmoid    -> C.Sigmoid    !>v
      | Activation.Tanh       -> C.Tanh       !>v
      | Activation.PReLU c    -> let alpha = new Constant(v.Output.Shape, dataType, c)
                                 C.PReLU(!>v, alpha)

        
