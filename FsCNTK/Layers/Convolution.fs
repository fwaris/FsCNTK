﻿namespace FsCNTK
open CNTK
open FsBase
open Blocks
open Layers

[<AutoOpen>]
module Layers_Convolution =
  type L with

    //** TODO move parameters out of function block

    static member Convolution
        (
            filter_shape,
            ?num_filters,
            ?activation,
            ?init,
            ?pad,
            ?strides,
            ?sharing,
            ?bias,
            ?init_bias,
            ?reduction_rank,
            ?dialation,
            ?groups,
            ?max_temp_mem_size_in_samples,
            ?name
        ) 
        =
        let num_filters = defaultArg num_filters 0
        let activation = defaultArg activation Activation.NONE
        let init = defaultArg init (C.GlorotUniformInitializer())
        let strides = defaultArg strides (D 1)
        let sharing = defaultArg sharing true
        let bias = defaultArg bias true
        let init_bias = defaultArg init_bias 0.
        let reduction_rank = defaultArg reduction_rank 1
        let dialation = defaultArg dialation (D 1)
        let groups = defaultArg groups 1
        let max_temp_mem_size_in_samples = defaultArg max_temp_mem_size_in_samples 0
        let name = defaultArg  name ""

        if [0;1] |> List.contains reduction_rank |> not then
            failwith "Convolution: reduction_rank must be 0 or 1"
        if not sharing then
            failwith "Convolution: sharing option currently must be True"
        if (groups <= 0) then
            failwith "Convolution: groups must be strictly positive, i.e. groups > 0."

        let out_channels = if num_filters = 0 then Ds[] else D num_filters
        let strd         = strides .padTo filter_shape
        let sharing      = asList (len filter_shape) sharing
        let dialation    = dialation .padTo filter_shape 

        let autoPadding = 
            let pd = 
                match pad with 
                | None-> asList ((len filter_shape) + reduction_rank) false 
                | Some p -> p
            if pd.Length = len filter_shape + 1 then 
                pd      // take padding as is as it includes the vector dimension
            elif pd.Length = len filter_shape then
               [false] @ pd
            elif pd.Length = 1 then 
               [false] @ [for _ in 1..len filter_shape -> List.head pd] //broadcast first element 
            else
               failwithf "padding if supplied, should be a single element list or one that has elements equal to filter dimensions (supplied: %A)" pad

        let filter_rank = len filter_shape

        fun (x:Node) ->

          let input_feature_map_depth = 
            if x |> O.shape |> len <= filter_rank then
              Ds []
            else
              (O.shape>>dims>>List.rev>>List.skip filter_rank>>List.rev>>Ds) x

          let kernel_shape = 
              out_channels
              + input_feature_map_depth
              + filter_shape

          let init_kernel = B._initializer_with_rank (init, filter_rank = filter_rank, output_rank = -1)

          let W = new Parameter(!--kernel_shape, dataType, init_kernel, device, "W")
          let b = 
              if bias then
                  new Parameter (
                        !-- (out_channels + (D 1) * filter_rank),
                        dataType, 
                        init_bias,
                        device,
                        "b")
              else
                  null
          let r = 
              C.Convolution (
                  W,
                  x.Var,
                  !--strd,
                  sharing |> List.rev |> boolVector,
                  autoPadding |> List.rev |> boolVector,
                  !--dialation,
                  uint32 reduction_rank,
                  uint32 groups,
                  uint32 max_temp_mem_size_in_samples
              )

          let r = if bias then C.Plus(!>r,b) else r

          let r = addActivation (F r) activation

          if !Layers.trace then printfn ">> Convolution[%s] %A" name (r |> O.shape |> dims)

          r
