namespace FsCNTK
open CNTK

//usually, much shape manipulation is done - so a separate type
type Shape = D (*size of dimension*) of int | Ds of int list | Unknown 
 
[<AutoOpen>]
module _shape_ = 
    let create_shape (dims:int seq) = NDShape.CreateNDShape dims

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
    member x.Item(i:int) = (dims x).[i] |> D

    member x.Dims = dims x

    member x.GetSlice(start1,finish1) = (x |> dims).GetSlice(start1,finish1) |> Ds

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
        | Ds is -> List.collect (fun x->x) [for _ in 1 .. repeat -> is] |> Ds
        | Unknown -> failwith "unspecified shape"

    member x.padTo (s2:Shape) =
        match x,s2 with
        | D i, D j -> D i
        | D i, Ds js -> js |> List.map (fun  _ -> i) |> Ds
        | Ds is, Ds js when is.Length=js.Length -> x
        | _,_ -> failwithf "shape must be singular or the dimensions should match s2"
