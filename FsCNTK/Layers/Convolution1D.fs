namespace FsCNTK
open FsBase
open Layers_Convolution

[<AutoOpen>]
module Layers_Convolution1D =
  type L with

    static member Convolution1D
        (
            filter_shape,
            ?num_filters,
            ?activation,
            ?init,
            ?pad,
            ?strides,
            ?bias,
            ?init_bias,
            ?reduction_rank,
            ?dialation,
            ?name
        ) 
        =
          if len(filter_shape) > 1 then failwith "Convolution1D: filter_shape must be a 1 D (scalar)"
          L.Convolution
            (
              filter_shape,
              num_filters   = defaultArg num_filters 0,
              activation    = defaultArg activation Activation.NONE,
              init          = defaultArg init (C.GlorotUniformInitializer()),
              pad           = defaultArg pad false,
              strides       = defaultArg strides (D 1),
              bias          = defaultArg bias true,
              init_bias     = defaultArg init_bias 0.,
              reduction_rank= defaultArg reduction_rank 1,
              dialation     = defaultArg dialation (D 1),
              name          = defaultArg name ""
            )

