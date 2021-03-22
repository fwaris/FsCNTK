namespace FsCNTK
open FsBase
open Layers_Convolution

[<AutoOpen>]
module Layers_Pooling =
  type L with

    static member private pad_to_shape (filter_shape:Shape, param:Shape, what) =
        let pS = len param
        let fS = len filter_shape
        if pS = 1 then //broadcast
            param .padTo filter_shape
        elif pS = fS then
            param
        else
            failwithf "%A parameter (%s) must be scaler (e.g. D 1) or have the same number of elements as filter shape %A" param what filter_shape
            
    static member Pooling
        (
            op:CNTK.PoolingType,
            filter_shape,
            ?strides,
            ?pad,
            ?name
        ) 
        =
        let strides = defaultArg strides (D 1)
        let name = defaultArg name ""

        let strides = L.pad_to_shape(filter_shape, strides, "strides")
        let pad = match pad with None -> asList (len filter_shape) false | Some p -> p
        
        fun (x:Node) ->
            C.Pooling(x.Var, op, !-- filter_shape, !-- strides, pad |> List.rev |> boolVector) |> F

    static member GlobalPooling
        (
            op:CNTK.PoolingType,
            ?name
        ) 
        =
        let name = defaultArg name ""
        L.Pooling(op, Shape.Unknown, name=name) 

    static member Unpooling
        (
            op:CNTK.PoolingType,
            filter_shape,
            ?strides,
            ?pad,
            ?name
        ) 
        =
        let strides = defaultArg strides (D 1)
        let name = defaultArg name ""

        let strides = L.pad_to_shape(filter_shape, strides, "strides")
        let pad = match pad with None -> asList (len filter_shape) false | Some p -> p
        
        fun (x:Node, y:Node) ->
            C.Unpooling(x.Var, y.Var, op, !-- filter_shape, !-- strides, pad |> List.rev |> boolVector) |> F

            