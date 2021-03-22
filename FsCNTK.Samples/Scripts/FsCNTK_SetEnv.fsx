#r @"..\..\packages\FSharp.Charting.2.1.0\lib\net45\FSharp.Charting.dll"
#r "System.Windows.Forms.DataVisualization"
open FSharp.Charting
module FsiAutoShow = 
    fsi.AddPrinter(fun (ch:FSharp.Charting.ChartTypes.GenericChart) -> ch.ShowChart() |> ignore; "(Chart)")

#r "netstandard" // without .dll seems to be the correct way
#r "nuget: FSharp.Charting"
#r "nuget: CNTK.GPU, Version=2.7.0"

let userProfile = System.Environment.GetEnvironmentVariable("UserProfile")
let packageRoot = $@"{userProfile}\.nuget\packages"
let nativeLib =  $@"{packageRoot}\cntk.gpu\2.7.0\support\x64\Release"
let path = System.Environment.GetEnvironmentVariable("path")
let path' =
    path
    + ";" + nativeLib
System.Environment.SetEnvironmentVariable("path",path')

#I @"..\..\FsCNTK.GPU"
#load "Probability.fs"
#load "FsBase.fs"
#load "Shape.fs"
#load "ValueInterop.fs"
#load "Node.fs"
#load "Operations.fs"
#load "Evaluation.fs"
#load "Training.fs"
#load "Blocks.fs"
#load "Layers\LayersBase.fs"
#load "Layers\Dense.fs"
#load "Layers\Dropout.fs"
#load "Layers\BatchNormalization.fs"
#load "Layers\LayerNormalization.fs"
#load "Layers\Convolution.fs"
#load "Layers\ConvolutionTranspose.fs"
#load "Layers\Pooling.fs"
#load "Layers\Sequence.fs"
#load "Layers\Attention.fs"

#load "..\ImageUtils.fs"
