namespace FsCNTK
open FsBase
open Layers_ConvolutionTranspose

[<AutoOpen>]
module Layers_ConvolutionTranspose2D =
  type L with
  //ConvolutionTranspose2D -- create a 2D convolution transpose layer with optional non-linearity
    static member ConvolutionTranspose2D 
        (
            filter_shape : Shape, //a 2D tuple, e.g., (3,3),
            ?num_filters,
            ?activation,
            ?init,
            ?pad,
            ?strides,
            ?bias,
            ?init_bias,
            ?output_shape,
            ?reduction_rank,
            ?dialation,
            ?name
        ) = 
               
        if len filter_shape  > 2 then            
            failwith "ConvolutionTranspose2D: filter_shape must be a scalar or a 2D tuple, e.g. 3 or (3,3)"

        let filter_shape = filter_shape .padTo (Ds [0;0])
        let output_shape = defaultArg output_shape Shape.Unknown

        L.ConvolutionTranspose(
            filter_shape,
            num_filters=defaultArg num_filters 0,
            activation=defaultArg activation Activation.NONE,
            init=defaultArg init (C.GlorotUniformInitializer()),
            pad=defaultArg pad false,
            strides=defaultArg strides (D 1),
            bias=defaultArg bias true,
            init_bias=defaultArg init_bias 0.,
            output_shape=output_shape,
            reduction_rank=defaultArg reduction_rank 1,
            dialation=defaultArg dialation (D 1),
            name=defaultArg name "")
