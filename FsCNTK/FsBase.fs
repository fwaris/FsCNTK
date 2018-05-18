namespace FsCNTK
open CNTK

// F# specific supporting and utility functions 

module FsBase =
  open System.Windows.Forms

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

  let schedule (ls : (int*float) seq) epochSize =
    ls 
    |> Seq.map (fun (n,r) ->  new PairSizeTDouble(uint32 n,r))
    |> ResizeArray
    |> (fun x -> new VectorPairSizeTDouble(x))
    |> (fun x  -> new TrainingParameterScheduleDouble(x, uint32 epochSize))

  let constSchedule (f:float) = new TrainingParameterScheduleDouble(f)

  let create_shape (dims:int seq) = NDShape.CreateNDShape dims

  //utility operator for F# implicit conversions 
  let inline (!>) (x:^a) : ^b = ((^a or ^b) : (static member op_Implicit : ^a -> ^b) x)

  let yourself x = x 

  //usually, much shape manipulation is done - so a separate type
 
  type Shape = D (*size of dimension*) of int | Ds of int list | Unknown 

  let dims = function D i -> [i] | Ds is -> is | Unknown -> failwith "unspecified shape"
  let len = function D i -> 1 | Ds is -> List.length is | Unknown -> 0
  let rev = function D i -> D i | Ds is -> List.rev is |> Ds | Unknown -> Unknown
  let private fromNDShape (s:NDShape) = s.Dimensions |> Seq.toList |> Ds
  let private ( !+ ) (s:NDShape) = fromNDShape s
  let private toNDShape = function D i -> create_shape [i] | Ds ds -> create_shape ds | Unknown -> NDShape.Unknown()
  let private ( !- ) s = toNDShape s
  //reverse order of shape dimensions to match .Net / C++ format which is column major
  let ( !-- ) s = s |> rev |> toNDShape
  let ( !++ ) (s:NDShape) = !+ s |> rev

  //Shape operations
  type Shape with 
      static member ( + ) (s1:Shape,s2:Shape) =
          match s1,s2 with
          | D i, D j -> Ds [i; j]
          | D i, Ds js -> List.append [i] js |> Ds
          | Ds is, D j -> List.append is [j] |> Ds
          | Ds is, Ds js -> List.append is js |> Ds
          | Unknown,_ 
          | _, Unknown -> failwith "unspecified shape"

      static member ( + ) (s1:Shape,d:int) =
          match s1 with
          | D i   -> Ds [i; d]
          | Ds is -> List.append is [d] |> Ds
          | Unknown -> failwith "unspecified shape"

      static member ( * )  (x:Shape, repeat:int) =
          match x with
          | D i -> Ds [for _ in 1 .. repeat -> i]
          | Ds is -> List.collect yourself [for _ in 1 .. repeat -> is] |> Ds
          | Unknown -> failwith "unspecified shape"

      member x.padTo (s2:Shape) =
          match x,s2 with
          | D i, D j -> D i
          | D i, Ds js -> js |> List.map (fun  _ -> i) |> Ds
          | Ds is, Ds js when is.Length=js.Length -> x
          | _,_ -> failwithf "shape must be singular or the dimensions should match s2"

  /// wrapper for CNTK Functions, Variables & Parameters:
  /// - for shape conversions (CNTK is column-major whereas Python API is row-major)
  /// - math operators
  type Node =
    | V of Variable
    | F of Function
    | P of Parameter
    with 

      member x.Var = match x with V v -> v | F f -> !> f | P p -> p :> Variable
      member x.Func = match x with V v -> v.ToFunction() | F f -> f | P p -> p.ToFunction()

      //probably not a good idea to use this directly
      //should use specfic helper methods
      static member private Variable (shape,?kind,?value,?needsGradient,?dynamicAxes,?isSparse,?name,?uid) =

        let kind          = defaultArg kind VariableKind.Input
        let value         = defaultArg value null
        let needsGradient = defaultArg needsGradient (kind=VariableKind.Input |> not)
        let dynamicAxes   = defaultArg dynamicAxes []
        let isSparse      = defaultArg isSparse false
        let name          = defaultArg name ""
        let uid           = defaultArg uid  (sprintf "%A%d" kind (System.DateTime.Now.ToFileTime()))

        let v             = new Variable(
                                  !-- shape,
                                  kind,
                                  dataType,
                                  value,
                                  needsGradient,
                                  dynamicAxes |> axisVector,
                                  isSparse,
                                  name,
                                  uid)
        V v

      static member Input (shape,?dynamicAxes,?isSparse,?needsGradient,?name) =

        let dynamicAxes   = defaultArg dynamicAxes []
        let isSparse      = defaultArg isSparse false
        let name          = defaultArg name ""
        let needsGradient = defaultArg needsGradient false

        let v             = Variable.InputVariable(
                                !-- shape,
                                dataType,
                                name,
                                ResizeArray dynamicAxes,
                                isSparse,
                                needsGradient)
        V v

      static member Placeholder(shape,dynamicAxes:Axis seq) =
        let v = Variable.PlaceholderVariable(!-- shape, ResizeArray(dynamicAxes))
        V v
                    
      static member Parm (shape,?init,?name) =
        let init = defaultArg init (C.GlorotUniformInitializer())
        let name = defaultArg name ""
        let W = 
            new Parameter(
                !-- shape,
                dataType,
                init,
                device, 
                name)
        P W

      static member Parm (shape,?init,?name) =
        let init = defaultArg init 0.
        let name = defaultArg name ""
        let W = 
            new Parameter(
                !-- shape,
                dataType,
                init,
                device, 
                name)
        P W

      static member private Const (c:float) = Constant.Scalar(dataType,c) 
      static member Scalar (c:float) = Node.Const c :> Variable |> V
   
      static member ( ./ ) (n:Node,d:float) = C.ElementDivide(n.Var, Node.Const d) |> F
      static member ( ./ ) (n:Node,d:Node) = C.ElementDivide(n.Var, d.Var) |> F

      static member ( .* ) (l:Node,r:float) = C.ElementTimes(l.Var, Node.Const r) |> F
      static member ( .* ) (l:float,r:Node) = C.ElementTimes(Node.Const l, r.Var) |> F
      static member ( .* ) (l:Node,r:Node) = C.ElementTimes(l.Var, r.Var) |> F

      static member ( * ) (l:Node,r:Node) = C.Times(r.Var,l.Var,"") |> F

      static member ( - ) (l:Node, r:Node) = C.Minus(l.Var, r.Var) |> F
      static member ( - ) (l:float, r:Node) = C.Minus(Node.Const l, r.Var) |> F
      static member ( - ) (l:Node, r:float) = C.Minus(l.Var, Node.Const r) |> F
      static member ( ~- ) (n:Node) = C.Negate(n.Var) |> F

      static member ( + ) (l:Node, r:Node) = C.Plus(l.Var, r.Var) |> F
      static member ( + ) (l:float, r:Node) = C.Plus(Node.Const l, r.Var) |> F
      static member ( + ) (l:Node, r:float) = C.Plus(l.Var, Node.Const r) |> F



