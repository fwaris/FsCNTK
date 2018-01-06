namespace FsCNTK
open CNTK
open System
open FsBase
open Blocks
open Layers

module Layers =
  type L with

    static member Convolution
        (
            filter_shape,
            ?num_filters,
            ?sequential,
            ?activation,
            ?init,
            ?pad,
            ?strides,
            ?sharing,
            ?bias,
            ?init_bias,
            ?reduction_rank,
            ?transpose_weight,
            ?dialation,
            ?max_temp_mem_size_in_samples,
            ?op_name,
            ?name
        ) 
        (convVar: Node)
        =
        let num_filters = defaultArg num_filters 0
        let sequential = defaultArg sequential false 
        let activation = defaultArg activation Activation.NONE
        let init = defaultArg init (C.GlorotUniformInitializer())
        let pad = defaultArg pad false
        let strides = defaultArg strides (D 1)
        let sharing = defaultArg sharing true
        let bias = defaultArg bias true
        let init_bias = defaultArg init_bias 0.
        let reduction_rank = defaultArg reduction_rank 1
        let transpose_weight = defaultArg transpose_weight false
        let dialation = defaultArg dialation (D 1)
        let max_temp_mem_size_in_samples = defaultArg max_temp_mem_size_in_samples 0
        let op_name = defaultArg op_name "Convolution"
        let name = defaultArg  name ""

        if [0;1] |> List.contains reduction_rank |> not then
            failwith "Convolution: reduction_rank must be 0 or 1"
        if transpose_weight then
            failwith "Convolution: transpose_weight option currently not supported"
        if not sharing then
            failwith "Convolution: sharing option currently must be True"

        let num_filters = if num_filters = 0 then Ds [] else D num_filters
        let filter_rank = len filter_shape
        let strides = strides .padTo filter_shape
        let sharing = asList filter_rank sharing 
        let pad     = asList filter_rank pad
        let dialation = dialation .padTo filter_shape

        let emulating_output_depth = len num_filters = 0
        let emulating_input_depth = reduction_rank = 0

        let actual_output_channels_shape = 
            if not emulating_output_depth then
                num_filters
            else
                D 1

        let actual_reduction_shape = D NDShape.InferredDimension 
        let actual_filter_shape = filter_shape

        let num_emulated_axes = if emulating_input_depth then 1 else 0
        let strides = (D 1) * num_emulated_axes + strides
        let sharing = asList num_emulated_axes true @ sharing
        let pad = asList num_emulated_axes false @ pad

        let kernel_shape = actual_reduction_shape + actual_filter_shape

        //simplified version of python code which I
        //don't fully understand yet
        let init_kernel = B._initializer_with_rank(
                            init,
                            filter_rank = filter_rank,
                            output_rank = -len(actual_output_channels_shape)
                            )

        let W = new Parameter(
                    !-(actual_output_channels_shape + kernel_shape),
                    dataType,
                    init_kernel,
                    device,
                    "W")

        let b = if bias then
                    new Parameter(
                        !-(actual_output_channels_shape + (D 1) * len(actual_filter_shape)),
                        dataType,
                        init_bias,
                        device,
                        "b")
                    else
                        null
            
        let filter_rank_without_seq = if sequential then filter_rank - 1 else filter_rank
        let num_inserted_axes = if sequential then 1 + num_emulated_axes else num_emulated_axes

        let beginAxis = 
            if filter_rank_without_seq <> 0 then 
                new Axis(-filter_rank_without_seq)
            else
                Axis.EndStaticAxis() //python code's Axis.new_leading_axis() resolves to this

        let endAxis = 
            if filter_rank_without_seq <> 0 then
                new Axis(-filter_rank_without_seq)
            else
                null

        let x = 
            if num_inserted_axes <> 0 then
                C.Reshape (
                    convVar.Var, 
                    !-- ((D 1) * num_inserted_axes),
                    beginAxis,
                    endAxis
                    )
            else
                !> convVar.Var

        let rank1 = (dims filter_shape |> List.rev).[filter_rank-1] //filter_shape[-filter_rank] in python
        let lpad = (rank1 - 1) / 2
        let x = 
            if sequential then
                let stride1 = (dims strides |> List.rev).[filter_rank-1]
                L._window(!>x,new Axis(-filter_rank),-lpad,-lpad+rank1,1,stride1)
            else
                x

        let sequential_emulated_axis = if sequential then pad.Length - filter_rank |> Some else None
        let isEmulated n = match sequential_emulated_axis with Some y -> y = n | None -> false
        let autoPadding =  
            asList reduction_rank false 
            @ 
            (pad |> List.mapi (fun i p -> if isEmulated i |> not then p else false))

        let r = C.Convolution(
                    W,
                    !>x,
                    !-- strides,
                    boolVector sharing,
                    boolVector autoPadding,
                    !-- dialation,
                    uint32 reduction_rank,
                    1u, //groups
                    uint32 max_temp_mem_size_in_samples,
                    "convolution")
            
        let zeroPad = (pad |> List.rev).[filter_rank - 1] 
        let r = 
            let begin_index = intVector [lpad]
            let end_index = intVector[-(rank1-1-lpad)]
            if sequential && not zeroPad then
                C.Slice(!>r, null, begin_index , end_index)
            else
                r

        let r = if bias then C.Plus(!>r,b) else r

        let num_axes_to_remove = [sequential; emulating_output_depth] |> List.map (function true -> 1 | false -> 0) |> List.sum
        let r = 
            if num_axes_to_remove > 0 then
                let begin_axis = new Axis(-filter_rank_without_seq - num_axes_to_remove)
                let end_axis = if filter_rank_without_seq <> 0 then new Axis(-filter_rank_without_seq) else null
                C.Reshape(!>r, !- (Ds []),  begin_axis, end_axis)
            else
                r
        let r = L.activation r activation

        r

