//TODO use Paket for reference management
(******** 
Note: This code is to reference dlls and native libs in
the CNTK nuget package. Adjust as needed for CPU or GPU versions

If this does not work, an alternative is to get the
the CNTK binary 'release' package from GitHub and 
save it to a local folder after extraction. 

Releases Page: https://github.com/Microsoft/CNTK/releases 

Put the directory that contains the binary files (dll's and libs)
on the windows Path. For the current iteration its the 'cntk' folder 
under the main extract.

Then reference the managed library - Cntk.Core.Managed-x.x.x.dll - from the same location for the script
and the lib compilation

The binary release package contains all native dlls in one place so its easier to access. 
Future nuget packages may move need packages around and so this code may become out of date.
*******)

open System
open System.IO

Environment.SetEnvironmentVariable("Path",
    Environment.GetEnvironmentVariable("Path") + ";" + __SOURCE_DIRECTORY__)

let pkgdir =  __SOURCE_DIRECTORY__ + @"..\..\..\packages"

let fullPath paths = Path.GetFullPath(Path.Combine(paths))

//change these for you installation (e.g. GPU vs CPU)
let dependencies = [
      @"CNTK.Deps.Cuda.2.5.0\support\x64\Dependency"
      @"CNTK.Deps.cuDNN.2.5.0\support\x64\Dependency"
      @"CNTK.Deps.MKL.2.5.0\support\x64\Dependency"
      @"CNTK.Deps.OpenCV.Zip.2.5.0\support\x64\Dependency"
      @"CNTK.Deps.OpenCV.Zip.2.5.0\support\x64\Dependency\Release"
      @"CNTK.GPU.2.5.0\support\x64\Release"
    ]

dependencies 
|> Seq.iter (fun dep -> 
    Environment.SetEnvironmentVariable("Path",
        fullPath [|pkgdir;dep|] + ";" + Environment.GetEnvironmentVariable("Path"))
    )    

#r @"..\..\packages\CNTK.GPU.2.5.0\lib\net45\x64\Cntk.Core.Managed-2.5.dll"

//for debugging from source
//debug output folder should be on path
//#I @"D:\Repos\cntk\x64\Debug"
//#r @"Cntk.Core.Managed-2.5d.dll"

//for release source build
//release output folder should be on path
//#r @"D:\Repos\cntk\x64\Debug\Cntk.Core.Managed-2.5.dll"

// Environment.GetEnvironmentVariable("Path").Split(';') |> Array.iter (printfn "%s")

//charting
#r @"..\..\packages\FSharp.Charting.0.91.1\lib\net45\FSharp.Charting.dll"
#r "System.Windows.Forms.DataVisualization"
open FSharp.Charting
module FsiAutoShow = 
    fsi.AddPrinter(fun (ch:FSharp.Charting.ChartTypes.GenericChart) -> ch.ShowChart() |> ignore; "(Chart)")


#load "..\ImageUtils.fs"
#load "..\Probability.fs"
#load "..\FsBase.fs"
#load "..\Operations.fs"
#load "..\Evaluation.fs"
#load "..\Blocks.fs"
#load "..\Layers\LayersBase.fs"
#load "..\Layers\Dense.fs"
#load "..\Layers\Dropout.fs"
#load "..\Layers\BatchNormalization.fs"
#load "..\Layers\Convolution.fs"
#load "..\Layers\Convolution2D.fs"
#load "..\Layers\ConvolutionTranspose.fs"
#load "..\Layers\ConvolutionTranspose2D.fs"
#load "..\Layers\Sequence.fs"
#load "..\Layers\Attention.fs"

