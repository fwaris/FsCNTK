namespace FsCNTK
open CNTK
open System
open FsBase
open Blocks
open Layers

module Layers =
  type L with

    static member ConvolutionTranspose 
        (
            convVar : Variable,
            filter_shape, 
            ?num_filters,
            ?activation,
            ?init,
            ?pad,
            ?strides,
            ?sharing,
            ?bias,
            ?init_bias,
            ?output_shape,
            ?reduction_rank,
            ?dialation,
            ?max_temp_mem_size_in_samples,
            ?name
        ) = 
        
        let num_filters = defaultArg num_filters 0 //probably not correct as python defaults to null
        let activation = defaultArg activation Activation.NONE
        let init = defaultArg init (C.GlorotUniformInitializer())
        let pad = defaultArg pad false
        let strides = defaultArg strides (D 1)
        let sharing = defaultArg sharing true
        let bias = defaultArg bias true
        let init_bias = defaultArg init_bias 0.
        let reduction_rank = defaultArg reduction_rank 1
        let dialation = defaultArg dialation (D 1)
        let max_temp_mem_size_in_samples = defaultArg max_temp_mem_size_in_samples 0
        let name = defaultArg  name ""

        if [0;1] |> List.contains reduction_rank |> not then
            failwith "ConvolutionTranspose: reduction_rank must be 0 or 1"
        if not sharing then 
            failwith "ConvolutionTranspose: sharing option currently must be true"

        //tuplify all tuple inputs that can also be given as scalars if rank 1
        //filter_shape = already given as Shape
        let num_filters  = D num_filters
        let strd1      = strides .padTo filter_shape
        let sharing      = asList (len filter_shape) sharing
        let pad          = asList (len filter_shape) pad 
        let dialation    = dialation .padTo filter_shape 

        let emulating_input_depth = if reduction_rank = 0 then 1 else 0

        let num_emulated_axes = emulating_input_depth
        let strd2 = (D 1) * num_emulated_axes + strd1
        let sharing = asList num_emulated_axes true @ sharing |> boolVector
        let pad     = asList num_emulated_axes false @ pad    
        let autoPadding = asList reduction_rank false @ pad |> boolVector
        let output_channels_shape = num_filters

        let kernel_shape = 
            D NDShape.InferredDimension
            + output_channels_shape
            + filter_shape

        let output_full_shape = 
            match output_shape with
            | None | Some Shape.Unknown -> output_channels_shape
            | Some (osp:Shape) -> output_channels_shape + osp

        let filter_rank = len filter_shape
        let init_kernel = B._initializer_with_rank (init, filter_rank = filter_rank, output_rank = -1)

        let W = new Parameter(!-kernel_shape, dataType, init_kernel,device,"W")
        let b = 
            if bias then
                new Parameter (
                    output_channels_shape + (D 1) * filter_rank |> toNDShape, 
                      dataType, 
                      init_bias,
                      device,
                      "b")
                  |> Some
            else
                None

        let num_inserted_axes = num_emulated_axes

        let beginAxis = 
            if filter_rank <> 0 then 
                new Axis(-filter_rank)
            else
                Axis.EndStaticAxis() //python code's Axis.new_leading_axis() resolves to this

        let endAxis = 
            if filter_rank <> 0 then
                new Axis(-filter_rank)
            else
                null

        let x = 
            if num_inserted_axes <> 0 then
                C.Reshape (
                    convVar, 
                    (D 1) * num_inserted_axes |> toNDShape,
                    beginAxis,
                    endAxis
                    )
            else
                !> convVar

        let r = 
            C.ConvolutionTranspose (
                W,
                !> x,
                !-strd2,
                sharing,
                autoPadding,
                !-output_full_shape,
                !-dialation,
                uint32 max_temp_mem_size_in_samples
            )

        let r = match b with Some b -> C.Plus(!>r,b) | None -> r

        L.activation r activation


