namespace FsCNTK
open CNTK
open System
open FsBase
open Blocks
open Layers

module Layers_BN =

  type L with
    static member BN
        (
              ?map_rank,
              ?init_scale,
              ?normalization_time_constant,
              ?blend_time_constant,
              ?epsilon,
              ?use_cntk_engine,
              ?disable_regularization,
              ?name
        ) 
        = 
        let map_rank =
            match map_rank with
            | None   -> 0
            | Some 1 -> 1
            | Some x -> failwith "map_rank can only be null or 1 for now"

        let normalization_time_constant = defaultArg normalization_time_constant 5000
        let blend_time_constant         = defaultArg blend_time_constant  0
        let epsilon                     = defaultArg epsilon  0.00001
        let use_cntk_engine             = defaultArg use_cntk_engine  false
        let init_scale                  = defaultArg init_scale 1.0
        let name                        = defaultArg name "batch norm."

        let norm_shape = !-- (D NDShape.InferredDimension)

        let scale        = new Parameter(norm_shape, dataType, init_scale, device, "scale")
        let bias         = new Parameter(norm_shape, dataType, 0., device, "bias")

        //comment in python code: # note: these are not really constants; they are updated differently
        //???
        let run_mean     = new Constant(norm_shape, dataType, 0., device, "aggregate_mean")
        let run_variance = new Constant(norm_shape, dataType, 0., device, "aggregate_variance")
        let run_count    = new Constant(!--(Ds []) , dataType, 0., device, "aggregate_count")

        fun (x:Node) -> 
            let r = 
                C.BatchNormalization 
                    (
                        x.Var,
                        scale,
                        bias,
                        run_mean,
                        run_variance,
                        run_count,
                        (map_rank = 1),
                        float normalization_time_constant,
                        float blend_time_constant,
                        epsilon,
                        not use_cntk_engine,
                        false,
                        name = name
                    )

            if !Layers.trace then printfn ">> BN[%s] %A" name r.Output.Shape.Dimensions

            F r

