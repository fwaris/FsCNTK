#load "SetEnv.fsx"
open FsCNTK
open FsCNTK.FsBase
open FsCNTK.Layers
open Layers_Dense
open Layers_BN
open Layers_ConvolutionTranspose2D
open Layers_Convolution2D
open CNTK
open System.IO

//Train a generative model using adversarial training to generate handwritten digits
//See this tutorial for background documentation: 
// https://cntk.ai/pythondocs/CNTK_206B_DCGAN.html

type C = CNTKLib
Layers.trace := true
let isFast = true
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

// the strides to be of the same length along each data dimension
let gkernel,dkernel =
    if kernel_h = kernel_w then
        kernel_h,kernel_h
    else
        failwith "This tutorial needs square shaped kernel"

let gstride,dstride =
    if stride_h = stride_w then
       stride_h, stride_h
    else
        failwith "This tutorial needs same stride in all dims"

let defaultInit() = C.NormalInitializer(0.2)

let bn_with_relu  = 
  L.BN (map_rank=1) 
  >> L.Activation Activation.ReLU

let bn_with_leaky_relu leak = 
  L.BN (map_rank=1) 
  >> L.Activation (Activation.PReLU leak)

//generator
let convolutional_generator =

  let s_h2, s_w2 = img_h / 2, img_w / 2 //Input shape (14,14)
  let s_h4, s_w4 = img_h / 4, img_w / 4 //Input shape (7,7)
  let gfc_dim = 1024
  let gf_dim = 64

  L.Dense (D gfc_dim, name="G h0")
  >> bn_with_relu
  >> L.Dense(Ds [gf_dim *2; s_h4; s_w4],init=defaultInit(), name="G h1")
  >> bn_with_relu
  >> L.ConvolutionTranspose2D
      (
        D gkernel,
        num_filters=gf_dim*2,
        strides=D gstride,
        pad=true,
        output_shape=Ds[s_h2; s_w2],
        name="G h2"
      )
  >> bn_with_relu
  >> L.ConvolutionTranspose2D
    (
      D gkernel,
      num_filters=1,
      strides=D gstride,
      pad=true,
      output_shape=Ds[img_h; img_w],
      name="G h3"
    )
  >> O.reshape (Ds [ g_output_dim])


// discriminator 
let convolutional_discriminator  =

  let dfc_dim = 1024
  let df_dim = 64

  O.reshape (Ds [1; img_h; img_w])
  >> L.Convolution2D(D dkernel, num_filters=1, strides=D dstride, name="D h0")
  >> bn_with_leaky_relu 0.2
  >> L.Convolution2D(D dkernel, num_filters=df_dim,strides=D dstride, name="D h1")
  >> bn_with_leaky_relu 0.2
  >> L.Dense (D dfc_dim, name = "D h2")
  >> bn_with_leaky_relu 0.2
  >> L.Dense(D 1, activation=Activation.Sigmoid, name="D=h3")

let minibatch_size = 128u
let num_minibatches = if isFast then 2 else 500000
let lr = 0.0002
let momentum = 0.5 //equivalent to beta1
let cntk_samples_folder = @"D:\Repos\cntk231\cntk\Examples\Image\DataSets\MNIST" //from CNTK download
//let cntk_samples_folder = @"F:\s\cntk\Examples\Image\DataSets\MNIST" //from CNTK download

let build_graph noise_shape image_shape generator discriminiator =
  let input_dynamic_axes = [Axis.DefaultBatchAxis()]
  let Z = Node.Input(noise_shape,dynamicAxes=input_dynamic_axes)

  let X_real = Node.Input(image_shape,dynamicAxes=input_dynamic_axes)
  let X_real_scaled = X_real ./ 255.0

  let X_fake = generator Z
  let D_real = discriminiator X_real_scaled

  let D_fake = D_real |> O.clone 
                ParameterCloningMethod.Share 
                (idict [O.outputVar X_real_scaled, O.outputVar X_fake])

             
  //loss functions generator and discriminator                
  let G_loss = 1.0 - O.log D_fake
  let D_loss = - (O.log D_real + O.log(1.0 - D_fake))

  G_loss.Func.Save(Path.Combine(@"D:\repodata\fscntk","G_loss.bin"))
  D_loss.Func.Save(Path.Combine(@"D:\repodata\fscntk","D_loss.bin"))
  D_fake.Func.Save(Path.Combine(@"D:\repodata\fscntk","D_fake.bin"))
  D_real.Func.Save(Path.Combine(@"D:\repodata\fscntk","D_real.bin"))

  let G_learner = C.AdamLearner(
                      O.parms X_fake |> parmVector,
                      new TrainingParameterScheduleDouble(lr,1u),
                      new TrainingParameterScheduleDouble(momentum))

  let D_learner = C.AdamLearner(
                      O.parms D_real |> parmVector,
                      new TrainingParameterScheduleDouble(lr,1u),
                      new TrainingParameterScheduleDouble(momentum))


  let G_trainer = C.CreateTrainer(X_fake.Func,G_loss.Func,null,lrnVector [G_learner])
  let D_trainer = C.CreateTrainer(D_real.Func,D_loss.Func,null,lrnVector [D_learner])

  X_real, X_fake, Z, G_trainer, D_trainer   

let streamConfigurations = 
    ResizeArray<StreamConfiguration>(
        [
            new StreamConfiguration(featureStreamName, imageSize)    
            new StreamConfiguration(labelsStreamName, numClasses)
        ]
        )

let minibatchSource = 
    MinibatchSource.TextFormatMinibatchSource(
        Path.Combine(cntk_samples_folder, "Train-28x28_cntk_text.txt"), 
        streamConfigurations, 
        MinibatchSource.InfinitelyRepeat)

let uniform_sample size =
    [|
        for _ in 1 .. size do
            let r = Probability.RNG.Value.NextDouble()
            yield  (float32 r - 0.5f) * 2.0f  //[-1,+1]
    |] 
    //uniform_sample 20

let noise_sample num_samples =
    let vals = uniform_sample  (num_samples * g_input_dim)
    let inp = Value.CreateBatch(create_shape [g_input_dim], vals, device)
    new MinibatchData(inp,uint32 minibatch_size)


let train (reader_train:MinibatchSource) generator discriminator =
    let X_real, X_fake, Z, G_trainer, D_trainer =
        build_graph 
            (D g_input_dim) 
            (D d_input_dim)
            generator
            discriminator
    
    let featureStreamInfo = reader_train.StreamInfo(featureStreamName)
    let k = 2 
    let print_frequency_mbsize = num_minibatches / 25
    //let pp_G = p()
    //let pp_D=  p()

    for train_step in 1 .. num_minibatches do

        //train the discriminator for k steps
        for gen_train_step in 1..k do
            let Z_data = noise_sample (int minibatch_size)
            let X_data = reader_train.GetNextMinibatch(minibatch_size)
            if X_data.[featureStreamInfo].numberOfSamples = Z_data.numberOfSamples then
                let batch_inputs = 
                    idict
                        [
                            X_real.Var, X_data.[featureStreamInfo]
                            Z.Var     , Z_data
                        ]
                D_trainer.TrainMinibatch(batch_inputs,device) |> ignore

        //train generator
        let Z_data = noise_sample (int minibatch_size)
        let batch_inputs = idict [Z.Var, Z_data]
        let b = G_trainer.TrainMinibatch(batch_inputs,device) //|> ignore 
        //hmmm python code does it twice

        if train_step % 100 = 0 then
            let l_D = D_trainer.PreviousMinibatchLossAverage()
            let l_G = G_trainer.PreviousMinibatchLossAverage()
            printfn "Minibatch: %d, D_loss=%f, G_loss=%f" train_step l_D l_G

    let G_trainer_loss = G_trainer.PreviousMinibatchLossAverage()
    Z, X_fake, G_trainer_loss


let reader_train = minibatchSource
//let d = reader_train.GetNextMinibatch(128u)
//d.Keys

//train model - can take a while even on a GPU if fast=false
let G_input, G_output, G_trainer_loss = train reader_train 
                                              convolutional_generator 
                                              convolutional_discriminator

// G_output.Func.Save(Path.Combine(@"D:\repodata\fscntk","Generator_Trained.bin"))

let noise = noise_sample 36
let outMap = idict[G_output.Func.Output,(null:Value)]
G_output.Func.Evaluate(idict[G_input.Var,noise.data],outMap,device)
let imgs = outMap.[G_output.Func.Output].GetDenseData<float32>(G_output.Func.Output)

let sMin,sMax,mid = 
    Seq.collect (fun x->x) imgs |> Seq.min, 
    Seq.collect (fun x->x) imgs |> Seq.max, 
    Seq.collect (fun x->x) imgs |> Seq.average

let grays = 
    imgs
    |> Seq.map (Seq.map (fun x-> if x < 0.f then 0uy else 255uy)>>Seq.toArray)
    //|> Seq.map (Seq.map (fun x -> Probability.scaler (0.,255.) (float sMin, float sMax) (float x) |> byte) >> Seq.toArray)
    |> Seq.map (ImageUtils.toGray (28,28))
    |> Seq.toArray

ImageUtils.showGrid (6,6) grays
(*
*)