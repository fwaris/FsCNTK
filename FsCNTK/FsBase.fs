namespace FsCNTK
open CNTK
type C = CNTKLib

// F# specific supporting and utility functions 

module FsBase =
  //let device = DeviceDescriptor.UseDefaultDevice()
  let device = DeviceDescriptor.GPUDevice(0) //should be configurable
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

  let create_shape (dims:int seq) = NDShape.CreateNDShape dims

  //utility operator for F# implicit conversions 
  let inline (!>) (x:^a) : ^b = ((^a or ^b) : (static member op_Implicit : ^a -> ^b) x)

  let yourself x = x 
  
  let scalar x = Constant.Scalar(dataType,x)

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

      member x.Var = match x with V v->v | F f -> !> f | P p -> !> p.ToFunction()
      member x.Func = match x with V v-> v.ToFunction() | F f -> f | P p -> p.ToFunction()

      static member CreateVar (shape,?kind,?value,?needsGradient,?dynamicAxes,?isSparse,?name,?uid) =

        let kind          = defaultArg kind VariableKind.Input        
        let value         = defaultArg value null
        let needsGradient = defaultArg needsGradient true
        let dynamicAxes   = defaultArg dynamicAxes []
        let isSparse      = defaultArg isSparse false
        let name          = defaultArg name ""
        let uid           = defaultArg uid  ""

        let v             = new Variable(
                                  !-- shape,
                                  kind,
                                  dataType,
                                  value,
                                  needsGradient,
                                  List.rev dynamicAxes |> axisVector,
                                  isSparse,
                                  name,
                                  uid)
        V v
      static member CreateParm (shape,?init,?name) =
        let init = defaultArg init (C.GlorotUniformInitializer())
        let name = defaultArg name "timesParam"
        let W = 
            new Parameter(
                !-- shape,
                dataType,
                init,
                device, 
                name)
        P W

  let shape = function
  | V v -> !++ v.Shape
  | F f -> !++ f.Output.Shape
  | P p -> !++ p.Shape

  let reshape shape (n:Node) = C.Reshape(n.Var, !-- shape) |> F

  let outputVar = function 
    | V v -> v
    | F f -> f.Output
    | P p -> p.ToFunction().Output

  let parms = function 
    | F f -> f.Parameters() 
    | V v -> v.ToFunction().Parameters()
    | P p -> p.ToFunction().Parameters()

  let clone method substitutions = function
    | V v -> v.ToFunction().Clone(ParameterCloningMethod.Share,substitutions) |> F
    | F f -> f.Clone(ParameterCloningMethod.Share,substitutions) |> F
    | P p -> p.ToFunction().Clone(ParameterCloningMethod.Share,substitutions) |> F

  let nLog (x:Node) = C.Log(x.Var) |> F

  type Node with 
    static member ( ./ ) (n:Node,d:float) = C.ElementDivide(n.Var, scalar d) |> F

    static member ( - ) (l:Node, r:Node) = C.Minus(l.Var, r.Var) |> F
    static member ( - ) (l:float, r:Node) = C.Minus(scalar l, r.Var) |> F
    static member ( - ) (l:Node, r:float) = C.Minus(l.Var, scalar r) |> F
    static member ( ~- ) (n:Node) = C.Negate(n.Var) |> F

    static member ( + ) (l:Node, r:Node) = C.Plus(l.Var, r.Var) |> F
    static member ( + ) (l:float, r:Node) = C.Plus(scalar l, r.Var) |> F
    static member ( + ) (l:Node, r:float) = C.Plus(l.Var, scalar r) |> F

