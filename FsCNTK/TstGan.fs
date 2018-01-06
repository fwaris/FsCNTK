module TstGan
open FsCNTK.FsBase
open FsCNTK.Layers
open CNTK
type C = CNTKLib

let featureStreamName = "features"
let labelsStreamName = "labels"
let imageSize = 28 * 28
let numClasses = 10

let img_h, img_w = 28, 28
let kernel_h, kernel_w = 5, 5
let stride_h, stride_w = 2, 2

let g_input_dim = 100
let g_output_dim = img_h * img_w

let d_input_dim = g_output_dim
let isFast = true

let s_h2, s_w2 = img_h / 2, img_w / 2 //Input shape (14,14)
let s_h4, s_w4 = img_h / 4, img_w / 4 //Input shape (7,7)
let gfc_dim = 1024
let gf_dim = 64

let convolution_generator (n:Node) =
  n
  |> Dense {pF with OutShape = D gfs_dim; Name="h0"}
  |> BN {pf with MapRank=1}
  |> Activation.ReLU
  |> Dense {pF with OutShape = Ds [gfc_dim * 2; s_h4; s_w4]; Name="h1"}
  |> BN {pf with MapRank=1}
  |> Activation.ReLU
  |> ConvTrans2D 
      {pf with 
        Kernel = D gkernel

      }