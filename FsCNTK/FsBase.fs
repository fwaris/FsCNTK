namespace FsCNTK
open CNTK

// F# specific supporting and utility functions 

[<AutoOpen>]
module FsBase =

  let trace = ref false

  type C = CNTKLib

  let inline asList sz x  = [for _ in 1 .. sz -> x]

  let inline idict (s:(^a * ^b) seq) =
      let d = new System.Collections.Generic.Dictionary< ^a, ^b>()
      s |> Seq.iter d.Add
      d

  //let device = DeviceDescriptor.UseDefaultDevice()
  let mutable device = DeviceDescriptor.GPUDevice(0) //should be configurable
  let dataType = DataType.Float

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

  let axisVector (is:Axis seq) =
      let vs = new AxisVector(Seq.length is)
      is |>  Seq.iter vs.Add
      vs


  //utility operator for F# implicit conversions 
  let inline (!>) (x:^a) : ^b = ((^a or ^b) : (static member op_Implicit : ^a -> ^b) x)

  let yourself x = x 




