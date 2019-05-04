namespace FsCNTK
open CNTK

//usually, much shape manipulation is done - so a separate type
type Shape = D (*size of dimension*) of int | Ds of int list  
#nowarn "40"
[<AutoOpen>]
module rec _shape_ = 
    let dims = function D i -> [i] | Ds is -> is 
    let len = function D i -> 1 | Ds is -> List.length is
    let rev = function D i -> D i | Ds is -> List.rev is |> Ds 
    let private fromNDShape (s:NDShape) = s.Dimensions |> Seq.toList |> Ds
    let private toNDShape = function D i -> create_shape [i] | Ds ds -> create_shape ds
    let private ( !+ ) (s:NDShape) = fromNDShape s
    let private unknown_shape = NDShape.Unknown()
    let private unkown_shape_dims = dims (!++ unknown_shape)
    let create_shape = function s when s = unkown_shape_dims -> unknown_shape | ds -> NDShape.CreateNDShape ds
    let private ( !- ) s = toNDShape s
    //reverse order of shape dimensions to match .Net / C++ format which is column major
    let ( !-- ) s = s |> rev |> toNDShape
    let  ( !++ ) (s:NDShape) = !+ s |> rev

//Shape operations
type Shape with 
    member x.Item(i:int) = (dims x).[i] |> D

    member x.Dims = dims x

    static member Unknown = !++ (NDShape.Unknown())

    member x.GetSlice(start1,finish1) = (x |> dims).GetSlice(start1,finish1) |> Ds

    static member ( + ) (s1:Shape,s2:Shape) =
        match s1,s2 with
        | D i, D j -> Ds [i; j]
        | D i, Ds js -> List.append [i] js |> Ds
        | Ds is, D j -> List.append is [j] |> Ds
        | Ds is, Ds js -> List.append is js |> Ds

    static member ( + ) (s1:Shape,d:int) =
        match s1 with
        | D i   -> Ds [i; d]
        | Ds is -> List.append is [d] |> Ds

    static member ( * )  (x:Shape, repeat:int) =
        match x with
        | D i -> Ds [for _ in 1 .. repeat -> i]
        | Ds is -> List.collect (fun x->x) [for _ in 1 .. repeat -> is] |> Ds

    member x.padTo (s2:Shape) =
        match x,s2 with
        | D i, D j -> D i
        | D i, Ds js -> js |> List.map (fun  _ -> i) |> Ds
        | Ds is, Ds js when is.Length=js.Length -> x
        | _,_ -> failwithf "shape must be singular or the dimensions should match s2"
