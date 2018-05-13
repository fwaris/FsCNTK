namespace FsCNTK
open FsBase
open CNTK
open System.Runtime.Remoting.Metadata.W3cXsd2001

//operation wrappers
type O =

  static member shape = function
    | V v -> !++ v.Shape
    | F f -> !++ f.Output.Shape
    | P p -> !++ p.Shape

  static member reshapeF shape (n:Node) = C.Reshape(n.Var, !-- shape) |> F //reshape for pipelining

  static member reshape (n:Node, shape:Shape, ?begin_axis, ?end_axis) = 
    let begin_axis = defaultArg begin_axis (new Axis 0)
    let end_axis = defaultArg end_axis (Axis.EndStaticAxis()) //python new_leading_axis resolve to this.
    C.Reshape(n.Var, !-- shape, begin_axis, end_axis) |> F

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

  static member log (x:Node) = C.Log(x.Var) |> F

  static member slice axis beginIndex endIndex (x:Node) = C.Slice (x.Var, axis, intVector beginIndex, intVector endIndex) |> F

  static member last (n:Node) = C.SequenceLast(n.Var) |> F

  static member combine (nodes:Node seq) = C.Combine(nodes |> Seq.map (fun n->n.Var) |> varVector) |> F
 
  static member uncombine (n:Node) = n.Func.Outputs |> Seq.map V |> Seq.toList
  
  static member mapOutputsZip (ys:(Node->Node) seq) =
    O.uncombine 
    >> Seq.zip ys 
    >> Seq.map (fun (a,b) -> a b) 
    >> O.combine

  static member mapOutputs (fn:(Node->Node)) =
    O.uncombine 
    >> Seq.map (fun x->fn x) 
    >> O.combine

  static member splice (ns:Node seq) = 
    C.Splice( ns |> Seq.map (fun n->n.Var) |> varVector, new Axis(0)) 
    |> F

  static member getOutput n = function 
    | F v when v.Outputs.Count < n - 1  -> failwithf "index exceeds avaiable output variables" 
    | F v                               -> v.Outputs.[n] |> V
    | _                                 -> failwith "for function nodes only"

  static member outputs (n:Node) = n.Func.Outputs |> Seq.map V |> Seq.toList

  static member sigmod (n:Node) = C.Sigmoid(n.Var) |> F

  static member softplus (n:Node) = C.Softplus(n.Var) |> F

  static member softmax (n:Node, ?axis, ?name ) = 
    let axis = defaultArg axis (new Axis(0))
    let name = defaultArg name ""
    C.Softmax(n.Var,axis,name) |> F

  static member reduce_sum(n:Node, ?axis, ?name) = 
    let axis = defaultArg axis (new Axis(0))
    let name = defaultArg name ""
    C.ReduceSum(n.Var, axis, name) |> F

  static member reduce_sum(n:Node, ?axes, ?name) = 
    let axes = defaultArg axes (axisVector [new Axis(0)])
    let name = defaultArg name ""
    C.ReduceSum(n.Var, axes, name) |> F

  static member squared_error (prediction:Node, targets:Node) = C.SquaredError(prediction.Var,targets.Var) |> F

  static member cross_entropy_with_softmax(z:Node,labels:Node) = C.CrossEntropyWithSoftmax(z.Var,labels.Var) |> F

  static member classification_error(z:Node,labels:Node) = C.ClassificationError(z.Var,labels.Var) |> F

  static member tanh (n:Node) = C.Tanh(n.Var) |> F

  static member times (l:Node, r:Node, ?output_rank, ?infer_input_rank_to_map, ?name) =
    let name = defaultArg name ""
    match output_rank, infer_input_rank_to_map with
    | Some o, Some i -> C.Times(r.Var, l.Var ,uint32 o,i, name) 
    | None, None     -> C.Times(r.Var,l.Var,name) 
    | _,_            -> failwith "output_rank and infer_input_rank_to_map should be either omitted or specified, together"
    |> F

  static member element_select(condition:Node, thenOperand:Node, elseOperand:Node) = 
    C.ElementSelect(condition.Var, thenOperand.Var, elseOperand.Var) |> F
    
  static member identity (n:Node)  = C.Combine(varVector [n.Var]) |> F

  static member seq_unpack (n:Node, padding_value, ?no_mask_output, ?name) =
    let no_mask_output = defaultArg no_mask_output false
    let name = defaultArg name ""
    C.SequenceUnpack(n.Var,padding_value,no_mask_output,name) |> F

  static member seq_broadcast_as (operand:Node, broadcast_as_operand:Node, ?name) =
    let name = defaultArg name ""
    C.SequenceBroadcastAs(operand.Var,broadcast_as_operand.Var,name) |> F

  static member seq_first (n:Node) = C.SequenceFirst(n.Var) |> F
  static member seq_last (n:Node) = C.SequenceLast(n.Var)   |> F