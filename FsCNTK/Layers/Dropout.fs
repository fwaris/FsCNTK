namespace FsCNTK
open CNTK
open System
open FsBase
open Blocks
open Layers

module Layers_Dropout =
  //this should be exposed in the C# Swig API but it is not
  //Python returns this value using the following code
  //from cntk.cntk_py import sentinel_value_for_auto_select_random_seed as SentinelValueForAutoSelectRandomSeed
  //seems to be:  System.UInt64.MaxValue - 2UL |> uint32 (from C++)
  let SentinelValueForAutoSelectRandomSeed = 4294967293L

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
        let seed = defaultArg seed C.SentinelValueForInferParamInitRank

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
      
