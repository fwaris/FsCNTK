namespace FsCNTK
open CNTK
open FsBase

module ValueInterop =
//utilty functions to create CNTK Value from raw data and vice-a-versa

  //handle default cases upto 3 dimensions (can't be empty)
  let toValue (o:obj) = 
    match o with
    | :? (float seq) as v   -> Value.CreateBatch(!--(D 1), v|> Seq.map float32, device)
    | :? (float32 seq) as v -> Value.CreateBatch(!--(D 1), v|> Seq.map float32, device)

    | :? ((float seq) seq) as v -> 
      let d = v |> Seq.head |>  Seq.length
      Value.CreateBatch(!--(D d), v |> Seq.collect yourself |> Seq.map float32, device)
    | :? ((float32 seq) seq) as v -> 
      let d = v |> Seq.head |>  Seq.length
      Value.CreateBatch(!--(D d), v |> Seq.collect yourself |> Seq.map float32, device)

    | :? (((float seq) seq) seq) as v -> 
      let d1 = v |> Seq.head |>  Seq.length
      let d2 = v |> Seq.head |> Seq.head |> Seq.length
      Value.CreateBatch(!--(Ds [d1;d2]), v |> Seq.collect (Seq.collect yourself) |> Seq.map float32, device)
    | :? (((float32 seq) seq) seq) as v -> 
      let d1 = v |> Seq.head |>  Seq.length
      let d2 = v |> Seq.head |> Seq.head |> Seq.length
      Value.CreateBatch(!--(Ds [d1;d2]), v |> Seq.collect (Seq.collect yourself) |> Seq.map float32, device)

    | x -> failwithf "value of type %A is not handled for implicit conversion to CNTK Value" x

  let toSeqValue (o:obj) =
    match o with
    | :? (float seq) as v ->  Value.CreateSequence(!--(D 1), v|> Seq.map float32, device)

    | :? ((float seq) seq) as v -> 
      let d = v |> Seq.head |>  Seq.length
      Value.CreateSequence(!--(D d), v |> Seq.collect yourself |> Seq.map float32, device)

    | :? (((float seq) seq) seq) as v -> 
      let d1 = v |> Seq.head |>  Seq.length
      let d2 = v |> Seq.head |> Seq.head |> Seq.length
      Value.CreateSequence(!--(Ds [d1;d2]), v |> Seq.collect (Seq.collect yourself) |> Seq.map float32, device)

    | x -> failwithf "value of type %A is not handled for implicit conversion to CNTK Value" x
 
  let getArray (v:Value) = 
    let ds = !++ v.Shape |> dims
    let var = Node.Input(Ds ds)
    v.GetDenseData<float32>(var.Var) |> Seq.map Seq.toArray |> Seq.toArray


    