namespace FsCNTK
open CNTK
open FsBase

[<AutoOpen>]
module Layers_LN =

  type L with
    static member LayerNormalization
        (
              ?initial_scale,
              ?initial_bias,
              ?epsilon,
              ?name
        ) 
        = 
        let initial_scale = defaultArg initial_scale 1.0
        let initial_bias = defaultArg initial_bias 0.
        let epsilon = defaultArg epsilon 0.00001
        let input_shape = D NDShape.InferredDimension 
        let scale = Node.Parm(input_shape, init=initial_scale)
        let bias = Node.Parm(input_shape,init=initial_bias)

        fun (x:Node) ->
            let mean = O.reduce_mean(x, new Axis(0))
            let x0 = x - mean
            let std = x0 |> O.square |> O.reduce_mean |> O.sqrt
            let std = 
                if epsilon <> 0.0 then
                    std + epsilon
                else
                    std
            let x' = x0 ./ std
            let r = (x' .* scale) + bias
            match name with None -> r | Some n -> O.alias(r,n)
