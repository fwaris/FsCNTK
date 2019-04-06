namespace FsCNTK
open CNTK
open System
open FsBase
open Blocks
open Layers


module Layers_Dense =

  type L with

    static member Dense
        (
            output_shape,
            ?activation,
            ?init,
            ?input_rank,
            ?map_rank,
            ?bias,
            ?init_bias,
            ?name
        )                            
        =
        let activation = defaultArg activation Activation.NONE
        let init = defaultArg init (C.GlorotUniformInitializer())
        let bias = defaultArg bias true
        let init_bias = defaultArg init_bias 0.
        let name = defaultArg  name ""

        let infer_input_rank_to_map =
            match input_rank,map_rank with
            | Some _, Some _    -> failwith "Dense: input_rank and map_rank cannot be specified at the same time."
            | Some _, None      -> -1
            | _     , None      -> 0
            | _     , Some r    -> r

        let output_rank = len output_shape
        let input_shape = D NDShape.InferredDimension * (defaultArg input_rank  1)
        let init_weight = B._initializer_with_rank (init, output_rank=output_rank) 
        let W = new Parameter(!--(input_shape + output_shape),dataType,init_weight,device,"W")
        let b = if bias then new Parameter(!--output_shape,dataType,init_bias,device,"b") else null

        fun (x:Node) ->
          let r = C.Times(W,x.Var,uint32 output_rank, infer_input_rank_to_map)
          let r = if bias then C.Plus(!>r,  b ) else r
          let r = addActivation (F r) activation
          if !Layers.trace then printfn ">> Dense[%s] %A" name (O.shape r) 
          r

