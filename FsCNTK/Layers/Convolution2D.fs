namespace FsCNTK
open CNTK
open System
open FsBase
open Blocks
open Layers
open Layers_Convolution

//based on python layers module (see CNTK Python API for documentation)
//mimics python code closely

module Layers_Convolution2D =
  type L with

    static member Convolution2D
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
          if len(filter_shape) > 2 then failwith "Convolution2D: filter_shape must be a scalar or a 2D tuple, e.g. 3 or (3,3)"
          let filter_shape = filter_shape .padTo (Ds [0;0])
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
