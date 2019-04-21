namespace FsCNTK
open FsBase

[<AutoOpen>]
module Layers_Dropout =

  type L with

    static member Dropout
        (
            ?dropout_rate,
            ?keep_prob,
            ?seed,
            ?name
        )                            
        =
        let name = defaultArg name ""
        let seed = defaultArg seed (C.GetRandomSeed())

        let dropout_rate = 
          match dropout_rate,keep_prob with
          | None  , None   -> failwith "Dropout: either dropout_rate or keep_prob must be specified."
          | Some _, Some _ -> failwith "Dropout: dropout_rate and keep_prob cannot be specified at the same time."
          | _     , Some k when  k < 0.0 || k >= 1.0 -> failwith "Dropout: keep_prob must be in the interval [0,1)"
          | _     , Some k -> 1.0 - k
          | Some d, _      -> d

        fun (x:Node) ->
          let r =  C.Dropout(x.Var, dropout_rate, uint32 seed, name=name)

          if !Layers.trace then printfn ">> Dropout[%f, %s] %A" dropout_rate name r.Output.Shape.Dimensions

          F r
      
