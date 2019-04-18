namespace FsCNTK
open CNTK
open FsBase
open Blocks
open Layers

[<AutoOpen>]
module Layers_ConvolutionTranspose =
  type L with

    //** TODO move parameters out of function block

    static member ConvolutionTranspose 
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
            ?output_shape,
            ?reduction_rank,
            ?dialation,
            ?max_temp_mem_size_in_samples,
            ?name
        ) 
        = 
        let num_filters = defaultArg num_filters 0 //scalar
        let activation = defaultArg activation Activation.NONE
        let init = defaultArg init (C.GlorotUniformInitializer())
        let pad = defaultArg pad false
        let strides = defaultArg strides (D 1)
        let sharing = defaultArg sharing true //for future
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

        let out_channels = if num_filters = 0 then Ds[] else D num_filters
        let strd         = strides .padTo filter_shape
        let sharing      = asList (len filter_shape) sharing
        let pad          = asList (len filter_shape) pad 
        let dialation    = dialation .padTo filter_shape 

        let autoPadding = asList reduction_rank false @ pad 
        let filter_rank = len filter_shape

        fun (x:Node) ->

          let input_feature_map_depth = 
            if x |> O.shape |> len <= filter_rank then
              Ds []
            else
              (O.shape>>dims>>List.rev>>List.skip filter_rank>>List.rev>>Ds) x

          let kernel_shape = 
              input_feature_map_depth
              + out_channels
              + filter_shape

          let output_full_shape = 
              match output_shape with
              | None | Some Shape.Unknown -> out_channels
              | Some (osp:Shape) -> out_channels + osp

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
              C.ConvolutionTranspose (
                  W,
                  x.Var,
                  !--strd,
                  sharing |> List.rev |> boolVector,
                  autoPadding |> List.rev |> boolVector,
                  !--output_full_shape,
                  !--dialation,
                  uint32 reduction_rank,
                  uint32 max_temp_mem_size_in_samples
              )

          let r = if bias then C.Plus(!>r,b) else r

          let r = addActivation (F r) activation

          if !Layers.trace then printfn ">> ConvolutionTranspose[%s] %A" name (r |> O.shape |> dims)

          r
