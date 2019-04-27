namespace FsCNTK
open CNTK
open System.Diagnostics


  /// wrapper for CNTK Functions, Variables & Parameters:
  /// - for shape conversions (CNTK is column-major whereas Python API is row-major)
  /// - math operators
 
  [<DebuggerDisplay("{DebugDisplay}")>]
  type Node =
    | V of Variable
    | F of Function
    | P of Parameter
    with 

      member x.Var = match x with V v -> v | F f -> !> f | P p -> p :> Variable
      member x.Func = match x with V v -> v.ToFunction() | F f -> f | P p -> p.ToFunction()
      member x.Shape = !++ x.Var.Shape

      member x.DebugDisplay = 
        let axisStr = (if x.Var.HasBatchAxis() then "#" else "") + (if x.Var.HasSequenceAxis() then ", *" else "")
        sprintf "%s %A [%s]" x.Var.Name ((!++ x.Var.Shape) |> dims) axisStr

      ///get output node from combined op output (not same as slicing)
      member x.Item(i:int) = 
        match x with 
        | F v when v.Outputs.Count < i - 1  -> failwithf "index exceeds avaiable output variables" 
        | F v                               -> v.Outputs.[i] |> V
        | x                                 -> x
     
      ///For simple slicing (use O.slice for complex slicing operations)
      member x.GetSlice(s1,e1) = 
        let axs = [Some (new Axis(0)) |> sanitize_axis] |> axisVector
        let s1 = defaultArg s1 0
        let e1 = match e1 with Some e -> e | None -> failwith "end index is required"
        C.Slice(x.Var, axs, intVector [s1], intVector [e1]) |> F

 
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

        let dynamicAxes   = defaultArg dynamicAxes [Axis.DefaultBatchAxis()]
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

      static member Placeholder(shape,?dynamicAxes:Axis seq) =
        let dynamicAxes = defaultArg dynamicAxes (Axis.UnknownDynamicAxes() |> Seq.cast<_>)
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

      static member private _const (c:float) =  Constant.Scalar(dataType, c, device)
      static member Scalar (c:float) = Node._const c :> Variable |> V
      static member Const (c:float) = new Constant( !> [| NDShape.InferredDimension |] , dataType, c) :> Variable |> V 
      static member Const (cs:NDArrayView) = new Constant(cs) :> Variable |> V

      static member Const (xs:float seq, ?shape) = 
        match shape with  
        | None   -> Node.Const(Vl.toValue(xs |> Seq.map float32).Data)
        | Some s ->  Node.Const(Vl.toValue(xs |> Seq.map float32, s).Data)
   
      static member ( ./ ) (n:Node,d:float) = C.ElementDivide(n.Var, Node._const d) |> F
      static member ( ./ ) (n:Node,d:Node) = C.ElementDivide(n.Var, d.Var) |> F

      static member ( .* ) (l:Node,r:float) = C.ElementTimes(l.Var, Node._const r) |> F  //Note: operators .* and ./ have lower precendence than + or - ; use parantheses to resolve precedence in mixed expressions
      static member ( .* ) (l:float,r:Node) = C.ElementTimes(Node._const l, r.Var) |> F
      static member ( .* ) (l:Node,r:Node) = C.ElementTimes(l.Var, r.Var) |> F

      static member ( * ) (l:Node,r:Node) = C.Times(r.Var,l.Var,"") |> F

      static member ( - ) (l:Node, r:Node) = C.Minus(l.Var, r.Var) |> F
      static member ( - ) (l:float, r:Node) = C.Minus(Node._const l, r.Var) |> F
      static member ( - ) (l:Node, r:float) = C.Minus(l.Var, Node._const r) |> F
      static member ( ~- ) (n:Node) = C.Negate(n.Var) |> F

      static member ( + ) (l:Node, r:Node) = C.Plus(l.Var, r.Var) |> F
      static member ( + ) (l:float, r:Node) = C.Plus(Node._const l, r.Var) |> F
      static member ( + ) (l:Node, r:float) = C.Plus(l.Var, Node._const r) |> F
