namespace FsCNTK
open CNTK
type C = CNTKLib

// F# specific supporting and utility functions 

module FsBase =
  let private create_shape (dims:int seq) = NDShape.CreateNDShape dims

  //utility operator for F# implicit conversions 
  let inline (!>) (x:^a) : ^b = ((^a or ^b) : (static member op_Implicit : ^a -> ^b) x)

  let yourself x = x 
  
  let scalar dataType x = Constant.Scalar(dataType,x)

  //usually, much shape manipulation is done - so a separate type
  type Shape = D (*size of dimension*) of int | Ds of int list | Unknown 
  let dims = function D i -> [i] | Ds is -> is | Unknown -> failwith "unspecified shape"
  let len = function D i -> 1 | Ds is -> List.length is | Unknown -> 0
  let rev = function D i -> D i | Ds is -> List.rev is |> Ds | Unknown -> Unknown
  let fromNDShape (s:NDShape) = s.Dimensions |> Seq.toList |> Ds
  let ( !+ ) (s:NDShape) = fromNDShape s
  let toNDShape = function D i -> create_shape [i] | Ds ds -> create_shape ds | Unknown -> NDShape.Unknown()
  let ( !- ) s = toNDShape s
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


  //type Parms =
  //  {
  //    Init          : unit->CNTKDictionary
  //    Activation    : Activation
  //    Strides       : Shape
  //    Dialation     : Shape
  //    Sharing       : bool list
  //    AutoPadding   : bool list
  //    LowerPad      : Shape
  //    UpperPad      : Shape
  //    Tranpose      : bool
  //    OutShape      : Shape
  //    Kernel        : Shape
  //    MaxTempMemSz  : int
  //    Axes          : Axis list
  //    BeginAxis     : Axis option
  //    EndAxis       : Axis option
  //    MapRank       : int
  //    Name          : string
  //  } with
  //  static member Default =
  //    {
  //      Init          = fun ()->C.GlorotNormalInitializer()
  //      Activation    = Activation.NONE
  //      Strides       = Shape.Unknown
  //      Dialation     = Shape.Unknown
  //      Sharing       = []
  //      AutoPadding   = []
  //      LowerPad      = Shape.Unknown
  //      UpperPad      = Shape.Unknown
  //      Tranpose      = false
  //      OutShape      = Shape.Unknown
  //      Kernel        = Shape.Unknown
  //      MaxTempMemSz  = 0
  //      Axes          = []
  //      BeginAxis     = None
  //      EndAxis       = None
  //      MapRank       = 1
  //      Name          = ""
  //  }

  //type VParms =
  //  {
  //    Kind          : VariableKind
  //    DataType      : DataType
  //    NeedsGradient : bool
  //    Value         : NDArrayView
  //    DynamicAxis   : Axis list
  //    IsSparse      : bool
  //  } with
  //    static member Default =
  //      {
  //        Kind          = VariableKind.Input
  //        DataType      = DataType.Float
  //        NeedsGradient = true
  //        DynamicAxis   = []
  //        Value         = null
  //        IsSparse      = false
  //      }

  //let pF = Parms.Default
  //let vF = VParms.Default

  type Node =
    | V of Variable
    | F of Function
    | P of Parameter
    with member x.Var = match x with V v->v | F f -> !> f | P _ -> failwith "not convertible"

  let shape = function
  | V v -> !++ v.Shape
  | F f -> !++ f.Output.Shape
  | P p -> !++ p.Shape
