﻿namespace FsCNTK
open FsBase
open CNTK

//operation wrappers
type O =

  static member shape = function
    | V v -> !++ v.Shape
    | F f -> !++ f.Output.Shape
    | P p -> !++ p.Shape

  static member pad(n:Node,pattern:(int*int) list, ?mode, ?constant_value) = 
    let mode = defaultArg mode PaddingMode.CONSTANTPAD
    let cnst = defaultArg constant_value 1.0
    let head = new SizeTVector()
    let foot = new SizeTVector()

    pattern 
    |> List.rev 
    |> List.iter (fun (h,t) -> head.Add(uint32 h); foot.Add(uint32 t))

    C.Pad(n.Var,mode,head,foot,cnst) |> F

  static member unpack_batch (n:Node, ?name) = C.UnpackBatch(n.Var,defaultArg name "") |> F

  static member to_batch(n:Node, ?name) = C.ToBatch(n.Var, defaultArg name "") |> F

  static member squeeze (n:Node, ?axes) = 
    match axes with
    | None -> C.Squeeze(n.Var)
    | Some axs -> C.Squeeze(n.Var, axs |> List.map (Some>>sanitize_axis) |>  axisVector)
    |> F

  static member reshapeF shape (n:Node) = C.Reshape(n.Var, !-- shape) |> F //reshape for pipelining

  static member reshape (n:Node, shape:Shape, ?begin_axis, ?end_axis) = 

    let begin_axis = defaultArg begin_axis (new Axis 0)
    let end_axis = defaultArg end_axis (Axis.EndStaticAxis()) //python new_leading_axis resolves to this.

    let sanitize_reshape_axis (a:Axis) =
      if not a.IsStatic then a
      elif a = Axis.EndStaticAxis() then new Axis(0)
      elif a = new Axis(0) then Axis.EndStaticAxis()
      else new Axis(- a.StaticAxisIndex())
    
    let internal_reshape_begin_axis = sanitize_reshape_axis(end_axis)
    let internal_reshape_end_axis = sanitize_reshape_axis(begin_axis)

    C.Reshape(n.Var, !-- shape, internal_reshape_begin_axis, internal_reshape_end_axis) |> F

  static member transpose (n:Node, ?axes) = 
    match axes with
    | None -> C.Transpose(n.Var)
    | Some axes -> C.Transpose(n.Var, axes |> sanitize_permutation |> axisVector)
    |> F

  static member reconcile_dynamic_axis (operand:Node, axesAsOperand:Node) = 
    C.ReconcileDynamicAxes(operand.Var, axesAsOperand.Var) |> F

  static member name = function V v -> v.Name | F f -> f.Name | P p -> p.Name

  static member outputVar = function 
    | V v -> v
    | F f -> f.Output
    | P p -> p.ToFunction().Output

  static member parms = function 
    | F f -> f.Parameters() 
    | V v -> v.ToFunction().Parameters()
    | P p -> p.ToFunction().Parameters()

  static member clone method substitutions = function
    | V v -> v.ToFunction().Clone(method,substitutions) |> F
    | F f -> f.Clone(method,substitutions) |> F
    | P p -> p.ToFunction().Clone(method,substitutions) |> F


  static member last (n:Node) = C.SequenceLast(n.Var) |> F

  static member combine (nodes:Node seq) = C.Combine(nodes |> Seq.map (fun n->n.Var) |> varVector) |> F
 
  static member uncombine (n:Node) = n.Func.Outputs |> Seq.map V |> Seq.toList

  static member OptimizedRNNStack(n:Node, w:Node, hiddenDim, ?layers, ?bidirectional, ?recurrentOp, ?name ) = 
    let layers = defaultArg layers 1
    let bidirectional = defaultArg bidirectional false
    let recurrentOp = defaultArg recurrentOp "lstm"
    let name = defaultArg name ""
    C.OptimizedRNNStack(n.Var, w.Var, uint32 hiddenDim,uint32 layers, bidirectional, recurrentOp, name) |> F
  
  static member mapOutputsZip (ys:(Node->Node) seq) =
    O.uncombine 
    >> Seq.zip ys 
    >> Seq.map (fun (a,b) -> a b) 
    >> O.combine

  static member mapOutputs (fn:(Node->Node)) =
    O.uncombine 
    >> Seq.map (fun x->fn x) 
    >> O.combine

  static member splice (ns:Node seq, ?axis) = 
    let axis = defaultArg axis (new Axis(-1))
    let axis = sanitize_axis (Some axis)
    C.Splice( ns |> Seq.map (fun n->n.Var) |> varVector, axis) 
    |> F
    
  static member slice (axes:AxisVector) =
    fun (beginIndex:int list) (endIndex:int list) (x:Node) ->
      C.Slice (x.Var, axes, intVector beginIndex, intVector endIndex) |> F

  static member slice (axes:Axis seq) =
    fun beginIndex endIndex (x:Node) ->
       O.slice (axisVector axes) beginIndex endIndex x

  static member slice (axes:int seq) =
    fun beginIndex endIndex (x:Node) ->
      let axes = axes |> Seq.map(fun (a:int) -> new Axis(a))
      O.slice axes beginIndex endIndex x

  static member getOutput n = function 
    | F v when v.Outputs.Count < n - 1  -> failwithf "index exceeds avaiable output variables" 
    | F v                               -> v.Outputs.[n] |> V
    | x                                 -> x
    //| _                               -> failwith "for function nodes only"

  static member outputs (n:Node) = n.Func.Outputs |> Seq.map V |> Seq.toList

  static member sigmod (n:Node) = C.Sigmoid(n.Var) |> F

  static member softplus (n:Node) = C.Softplus(n.Var) |> F

  static member pow (n:Node, p:Node) = C.Pow(n.Var,p.Var) |> F

  static member clip(n:Node,l:Node,h:Node) = C.Clip(n.Var,l.Var,h.Var) |> F

  static member mean (n:Node) = C.Mean(varVector [n.Var]) |> F

  static member max (n1:Node, n2:Node) = C.ElementMax(n1.Var, n2.Var, "max") |> F

  static member min (n1:Node, n2:Node) = C.ElementMin(n1.Var, n2.Var, "nin") |> F

  static member eq (n1:Node, n2:Node) = C.Equal(n1.Var,n2.Var) |> F

  static member log (x:Node) = C.Log(x.Var) |> F

  static member exp(n:Node) = C.Exp(n.Var) |> F

  static member abs(n:Node) = C.Abs(n.Var) |> F

  static member sum(n:Node seq) = C.Sum( n |> Seq.map (fun x->x.Var) |> varVector) |> F

  static member flatten(n:Node, ?axis, ?name) =
    match axis with
    | None -> C.Flatten(n.Var)
    | Some a -> 
        let a = sanitize_axis a
        C.Flatten(n.Var,a,defaultArg name "")
    |> F

  static member reduce_mean(n:Node, ?axis, ?name) = 
    C.ReduceMean(n.Var, sanitize_axis axis, defaultArg name "") |> F

  static member reduce_mean(n:Node, axes, ?keepDims, ?name) = 
    C.ReduceMean(n.Var, axes |> sanitize_multi_axis_reduction_list |> axisVector, defaultArg keepDims  true, defaultArg name "") |> F

  static member softmax (n:Node, ?axis, ?name ) = 
    C.Softmax (n.Var, sanitize_axis axis, defaultArg name "") |> F

  static member reduce_sum(n:Node, ?axis, ?name) = 
    C.ReduceSum(n.Var, sanitize_axis axis, defaultArg name "") |> F

  static member reduce_sum(n:Node, axes, ?keepDims, ?name) = 
    C.ReduceSum(n.Var, axes |> sanitize_multi_axis_reduction_list |> axisVector, defaultArg keepDims  true, defaultArg name "") |> F

  static member reduce_max(n:Node, ?axis, ?name) = 
    C.ReduceMax(n.Var, sanitize_axis axis, defaultArg name "") |> F

  static member reduce_max(n:Node, axes, ?keepDims, ?name) = 
    C.ReduceMax(n.Var, axes |> sanitize_multi_axis_reduction_list |> axisVector, defaultArg keepDims  true, defaultArg name "") |> F

  static member random_normal shape = C.NormalRandom(!-- shape, dataType) |> F //zero mean, unit variance ?

  static member random_normal(shape,mu,sigma, ?seed, ?name) = 
    match seed with 
    | None -> C.NormalRandom(!-- shape,dataType,mu,sigma) |> F  //not sure about the default seed used so let that default to base API
    | Some s -> C.NormalRandom(!-- shape,dataType,mu,sigma,uint32 s,defaultArg name "") |> F

  static member uniform(shape,?low,?high, ?seed, ?name) = 
    let low = defaultArg low 0.
    let high = defaultArg high 1.0
    let name = defaultArg name ""
    let seed = defaultArg seed (C.GetRandomSeed())
    C.UniformRandom(!-- shape, dataType,low,high,uint32 seed, name) |> F

  static member uniform_like(n:Node,?low,?high, ?seed, ?name) = 
    let low = defaultArg low 0.
    let high = defaultArg high 1.0
    let name = defaultArg name ""
    let seed = defaultArg seed (C.GetRandomSeed())
    C.UniformRandomLike(n.Var,low,high,uint32 seed, name) |> F

  ///radomly permute the columns across the minibatch
  static member randperm (x:Node, minibatch_size) =
    let x' = O.unpack_batch x //remove dynamic axis

    if x'.Shape.Dims.Length <> 2 then failwith "only 2D tensors supported"
    //can use flatten here?

    let rows = minibatch_size       //this has to be statically given
    let cols = x'.Shape.Dims.[1]
    let dataShape = Ds [rows; cols]

    let rands = O.uniform(dataShape,1.,float rows) 
    let idxs = (O.round rands) - Node.Scalar(1.0) // 0 to (rows-1)
    let idx_rowOffsets = idxs .* (float cols)
    let colOffsets = Node.Const([for i in 0. .. float(cols-1) -> i], D cols)

    let idxs_colOffsets = idx_rowOffsets + colOffsets   //gather used below supports 1 axis
                                                        //this corrects the indices so that the corresponding
                                                        //col is used from the (randomly) referenced row
    let flatDim = rows * cols
    let flatX = O.reshape(x', D flatDim)      //1D rows
    let flatIdxs = O.reshape(idxs_colOffsets, D flatDim)
    let flatXPerm = O.gather(flatX,flatIdxs, new Axis(0))   //gather using the indices created
    let xperm = O.reshape(flatXPerm, dataShape)
    let x'' = O.to_batch xperm
    x''

  static member round (n:Node) = C.Round(n.Var) |> F

  static member hardmax (n:Node, ?name) = C.Hardmax(n.Var, defaultArg name "") |> F

  static member squared_error (prediction:Node, targets:Node, ?name) = 
    C.SquaredError(prediction.Var,targets.Var, defaultArg name "" ) |> F

  static member square (n:Node, ?name) = C.Square(n.Var, defaultArg name "") |> F

  static member sqrt (n:Node, ?name) = C.Sqrt(n.Var, defaultArg name "") |> F

  static member cross_entropy_with_softmax(z:Node,labels:Node, ?axis, ?name) = 
    match axis with 
    | None -> C.CrossEntropyWithSoftmax(z.Var,labels.Var,defaultArg name "") |> F
    | Some axis -> 
        let axis = axis |> Some |> sanitize_axis
        C.CrossEntropyWithSoftmax(z.Var,labels.Var, axis) |> F

  static member lambda_rank(output:Node, gain:Node, group:Node) =
    C.LambdaRank(output.Var, gain.Var, group.Var) |> F


  static member binary_cross_entropy (z:Node, labels:Node,?name) = C.BinaryCrossEntropy(z.Var,labels.Var, defaultArg name "") |> F

  static member classification_error(z:Node,labels:Node, ?name) = C.ClassificationError(z.Var,labels.Var, defaultArg name "") |> F

  static member tanh (n:Node) = C.Tanh(n.Var) |> F

  static member gather(reference:Node, indices:Node, ?axis, ?name) =
    match axis with
    | None -> C.GatherOp(indices.Var, reference.Var) |> F
    | Some ax -> 
        let ax = sanitize_axis (Some ax)
        C.GatherOp(indices.Var,reference.Var, ax, defaultArg name "") |> F

  static member element_select(condition:Node, thenOperand:Node, elseOperand:Node, ?name) = 
    C.ElementSelect(condition.Var, thenOperand.Var, elseOperand.Var, defaultArg name "") |> F

  static member times (l:Node, r:Node, ?output_rank, ?infer_input_rank_to_map, ?name) =
    let name = defaultArg name ""
    match output_rank, infer_input_rank_to_map with
    | Some o, Some i -> C.Times(r.Var, l.Var ,uint32 o,i, name) 
    | None, None     -> C.Times(r.Var,l.Var,name) 
    | _,_            -> failwith "output_rank and infer_input_rank_to_map should be either omitted or specified, together"
    |> F

    
  static member identity (n:Node)  = C.Combine(varVector [n.Var]) |> F //can't we just return the node?

  static member zeros_like (n:Node) = C.ZerosLike(n.Var) |> F

  static member ones_like (n:Node) = C.OnesLike(n.Var) |> F

  static member alias (n:Node, ?name) = C.Alias(n.Var,defaultArg name "") |> F

  static member as_composite name (n:Node) = C.AsComposite(n.Func,name) |> F

  static member seq_unpack (n:Node, padding_value, ?no_mask_output, ?name) =
    let no_mask_output = defaultArg no_mask_output false
    let name = defaultArg name ""
    C.SequenceUnpack(n.Var,padding_value,no_mask_output,name) |> F

  static member seq_broadcast_as (operand:Node, broadcast_as_operand:Node, ?name) =
    let name = defaultArg name ""
    C.SequenceBroadcastAs(operand.Var,broadcast_as_operand.Var,name) |> F

  static member seq_where (n:Node) = C.SequenceWhere(n.Var) |> F

  static member seq_first (n:Node) = C.SequenceFirst(n.Var) |> F
  static member seq_last (n:Node) = C.SequenceLast(n.Var)   |> F

  static member seq_past_value (n:Node,?time_step:int) =  
    C.PastValue(n.Var,  defaultArg time_step 1 |> uint32) |> F

  static member seq_future_value (n:Node,?time_step:int) =  
    C.FutureValue(n.Var,  defaultArg time_step 1 |> uint32) |> F

  static member seq_gather (operand:Node, condition:Node, ?name) = 
    let name = defaultArg name ""
    C.SequenceGather(operand.Var, condition.Var, name) |> F

  static member seq_scatter (operand:Node, condition:Node, ?name) = 
    let name = defaultArg name ""
    C.SequenceScatter(operand.Var, condition.Var, name) |> F

  static member seq_is_first (n:Node, ?name) = C.SequenceIsFirst(n.Var, defaultArg name "") |> F
  static member seq_is_last (n:Node, ?name) = C.SequenceIsLast(n.Var,  defaultArg name "") |> F

  static member seq_slice (n:Node, begIdx, endIdx, ?name) = 
    C.SequenceSlice(n.Var,begIdx,endIdx,  defaultArg name "") |> F


//extensions for Node that leverage operations defined above
