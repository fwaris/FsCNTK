namespace FsCNTK
open CNTK

type Vl =
//utilty functions to create CNTK Value from raw data and vice-a-versa

  //handle default cases upto 3 dimensions (can't be empty)
  static member toValue (v:seq<float>, ?shape) =  
    let shape = defaultArg shape (D 1)
    Value.CreateBatch(!-- shape, v|> Seq.map float32, device)

  static member toValue (v:seq<float32>, ?shape) =  
    let shape = defaultArg shape (D 1)
    Value.CreateBatch(!-- shape, v , device)

  static member toSeqValue (v:#seq<float>, ?shape)  =
    let shape = defaultArg shape (D 1)
    Value.CreateSequence(!-- shape, v|> Seq.map float32, device)

  static member toSeqValue (v:#seq<#seq<float>>, ?shape)  =
    let d = v |> Seq.head |>  Seq.length
    let shape = defaultArg shape (D d)
    Value.CreateSequence(!-- shape, v  |> Seq.collect (Seq.map float32), device)

  static member toSeqValue (v:#seq<#seq<#seq<float>>>, ?shape)  =
    let d1 = v |> Seq.head |>  Seq.length
    let d2 = v |> Seq.head |> Seq.head |> Seq.length
    let shape = defaultArg shape (Ds [d1;d2])
    Value.CreateSequence(!-- shape, v |> Seq.collect (Seq.collect (Seq.map float32)), device)

  static member toSeqBatches (v:#seq<#seq<float>>, ?shape)  =
    let d = v |> Seq.head |>  Seq.length
    let shape = defaultArg shape (D d)
    Value.CreateBatchOfSequences(!-- shape, v  |> Seq.map (Seq.map float32), device)

  static member toSeqBatches (v:#seq<#seq<#seq<float>>>, ?shape)  =
    let d = v |> Seq.head |> Seq.head |> Seq.length
    let shape = defaultArg shape (D d)
    let vs = v |> Seq.map (Seq.collect (Seq.map float32))
    Value.CreateBatchOfSequences(!-- shape, vs, device)
     
  static member getArray (v:Value) = 
    use v = v.DeepClone(true)
    let xs = v.GetDenseData<float32>(new Constant(v.Data) :> Variable) |> Seq.map Seq.toArray |> Seq.toArray |> Array.head
    v.Dispose()
    xs

  static member getArray(data:NDArrayView) =
    let v = new Value(data)
    Vl.getArray(v)

  static member parameterNames(f:CNTK.Function) =
    f.Parameters()
    |> Seq.map(fun x-> x.Uid, x.Name)
    |> Seq.toArray
 
  static member parmVal (f:CNTK.Function,uid:string) =
    f.Parameters()
    |> Seq.tryFind(fun x->x.Uid=uid)
    |> Option.map(fun x->x.Shape, x.Value() |> Vl.getArray)
        
   
                                       


    