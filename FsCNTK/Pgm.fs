module Pgm

open System
open System.IO
open System.Collections.Generic

open CNTK
open CNTKWrapper.FsBase
open CNTKWrapper.Blocks
open CNTKWrapper.Layers
type C = CNTKLib
open ImageUtils

let input_shape  = Ds [ 128; 7; 7]
let kernel_shape = Ds [NDShape.InferredDimension; 128; 5; 5]
let output_shape = Ds [128; 14; 14]
let strides      = Ds [2; 2]
let dialation    = Ds [1; 1]
let sharing      = [true; true]
let padding      = [false; true; true]

let input_shape'  = rev input_shape
let kernel_shape' = rev kernel_shape
let output_shape' = rev output_shape 
let strides'      = rev strides 
let dialation'    = rev dialation
let sharing'      = List.rev sharing
let padding'      = List.rev padding

let init = C.GlorotUniformInitializer()
let init_kernel = B._initializer_with_rank(init , filter_rank = len kernel_shape', output_rank = -1)

let W = new Parameter(!-kernel_shape', dataType, init_kernel,device,"W")
let dx = new AxisVector([|Axis.DefaultDynamicAxis()|])
let x = new Variable(!-input_shape', VariableKind.Input, dataType, null, true, dx, false, "","1")
W.Shape.Dimensions
x.Shape.Dimensions

//let r1 = C.ConvolutionTranspose(W,x)
//r1.Output.Shape.Dimensions
let r = 
  C.ConvolutionTranspose (
      W,
      x,
      !-strides',                        
      boolVector sharing',          
      boolVector padding',    
      !-output_shape',                   
      !-dialation',
      1u,                                //reduction_rank
      0u                                 //max_temp_mem_size_in_samples
  )
 