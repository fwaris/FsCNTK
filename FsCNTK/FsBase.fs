namespace FsCNTK
open CNTK

// F# specific supporting and utility functions 

[<AutoOpen>]
module FsBase =

  let trace = ref false

  type C = CNTKLib

  //this should be exposed in the C# Swig API but it is not
  //Python returns this value using the following code
  //from cntk.cntk_py import sentinel_value_for_auto_select_random_seed as SentinelValueForAutoSelectRandomSeed
  //seems to be:  System.UInt64.MaxValue - 2UL |> uint32 (from C++)
  let SentinelValueForAutoSelectRandomSeed = 4294967293L

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

  let sanitize_axis (a:Axis option) =
      match a with 
      | None    -> Axis.AllStaticAxes()
      | Some a when a.IsStatic && a.StaticAxisIndex() <> Axis.EndStaticAxis().StaticAxisIndex() 
                -> new Axis(-1 - a.StaticAxisIndex())
      | Some a  -> a

  let sanitize_multi_axis_reduction_list (a:Axis list option) =
      match a with 
      | None  -> [Axis.AllStaticAxes()]
      | Some axs when axs |> List.exists (fun a->a.IsSequenceAxis()) -> failwith "Reduction operation over multiple axes can not contain sequence axis"
      | Some axs -> axs |> List.map (Some>>sanitize_axis)

  let sanitize_permutation (perm:Axis list) =
      let axs = perm |> List.map (fun a->a.StaticAxisIndex())
      let axs' = List.distinct axs 
      if axs.Length <> axs'.Length then failwith "duplicate axis detected"
      match perm |> List.tryFind (fun a -> a.StaticAxisIndex() >= -perm.Length && a.StaticAxisIndex() < perm.Length |> not) with
      | Some a -> failwithf "invalid axis %A: elements must be from -%d to %d" a -perm.Length (perm.Length - 1)
      | None -> perm |> List.rev |> List.map (fun a-> new Axis(perm.Length - a.StaticAxisIndex() - 1))

  //utility operator for F# implicit conversions 
  let inline (!>) (x:^a) : ^b = ((^a or ^b) : (static member op_Implicit : ^a -> ^b) x)

  let yourself x = x 




