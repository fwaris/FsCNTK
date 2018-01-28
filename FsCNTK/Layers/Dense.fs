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

        fun (x:Node) ->

          //python uses late binding so shape is inferred
          //here we can just use the shape given
          let input_shape = O.shape x
            
          let init_weight = B._initializer_with_rank (init, output_rank=output_rank) 
          let W = Node.Parm(input_shape + output_shape, init=init_weight,name="W") 
          let b = if bias then Node.Parm(output_shape,init=init_bias,name="b") |> Some else None

          //python code swaps left and right parameters in its times function - don't know why
          //here we use the cntk function 
          let r = O.times2(x,W, output_rank, infer_input_rank_to_map,"")
          let r = match b with None -> W | Some b -> b + W
          let r = addActivation r activation

          if !Layers.trace then printfn ">> Dense[%s] %A" name (r|>O.shape|>dims)

          r
      
