(*
This file is intended to load dependencies in an F# script,
to train a model from the scripting environment.
CNTK, CPU only, is assumed to have been installed via Paket.
*)

open System
open System.IO

Environment.SetEnvironmentVariable("Path",
    Environment.GetEnvironmentVariable("Path") + ";" + __SOURCE_DIRECTORY__)

//change it for you installation (e.g. GPU vs CPU)
let baseDir = __SOURCE_DIRECTORY__ + @"..\..\..\packages\CNTK.GPU.2.3.1\"
let dependencies = [
        Path.GetFullPath(Path.Combine(baseDir,"lib/net45/x64/"))
        Path.GetFullPath(Path.Combine(baseDir,"support/x64/Release/"))
        Path.GetFullPath(Path.Combine(baseDir,"support/x64/Dependency/Release/"))
        Path.GetFullPath(Path.Combine(baseDir,"support/x64/Release/"))
        //@"D:\Repos\cntk231\cntk\x64\Debug"
    ]


dependencies 
|> Seq.iter (fun dep -> 
    let path = Path.Combine(__SOURCE_DIRECTORY__,dep)
    Environment.SetEnvironmentVariable("Path",
        path + ";" + Environment.GetEnvironmentVariable("Path"))
    )    

//#I @"../../packages/CNTK.GPU.2.3.1/lib/net45/x64/"
//#I @"../../packages/CNTK.GPU.2.3.1/support/x64/Dependency/"
//#I @"../../packages/CNTK.GPU.2.3.1/support/x64/Dependency/Release/"
//#I @"../../packages/CNTK.GPU.2.3.1/support/x64/Release/"


#r @"..\..\packages\CNTK.GPU.2.3.1\lib\net45\x64\Cntk.Core.Managed-2.3.1.dll"
//#r @"D:\Repos\cntk231\cntk\x64\Debug\Cntk.Core.Managed-2.3.1d.dll"



#load "..\ImageUtils.fs"
#load "..\Probability.fs"
#load "..\FsBase.fs"
#load "..\Blocks.fs"
#load "..\Layers\LayersBase.fs"
#load "..\Layers\Dense.fs"
#load "..\Layers\BatchNormalization.fs"
#load "..\Layers\Convolution.fs"
#load "..\Layers\Convolution2D.fs"
#load "..\Layers\ConvolutionTranspose.fs"
#load "..\Layers\ConvolutionTranspose2D.fs"

