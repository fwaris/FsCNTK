namespace FsCNTK
open CNTK
open System
open FsBase

//based on python blocks module (see CNTK Python API for documentation)

[<AutoOpen>]
module Blocks =

    type B =

        static member _initializer_with_rank (init, ?output_rank, ?filter_rank) =
            let output_rank = output_rank |> Option.defaultValue C.SentinelValueForInferParamInitRank
            let filter_rank = filter_rank |> Option.defaultValue C.SentinelValueForInferParamInitRank
            C.RandomInitializerWithRank(init, output_rank, filter_rank)

        static member Stabilizer
          (
            ?steepness,
            ?enable_self_stabilization,
            ?name
          )
          =
            let steepness = defaultArg steepness 4
            let enable_self_stabilization = defaultArg enable_self_stabilization true
            let name = defaultArg name ""
            let init_parm = log(exp(float steepness) - 1.0) / (float steepness)
            let param = Node.Parm(Ds[],init=init_parm,name="alpha")
            let param = if steepness = 1 then param else (float steepness) .* param 
            let beta = O.softplus param 

            fun (x:Node) ->
              if not enable_self_stabilization then x
              else
                let r = beta .* x
                r


