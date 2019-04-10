#load "SetEnv.fsx"
open FsCNTK
open FsCNTK.FsBase
open Layers_Dense
open CNTK
open System.IO
open System
open CNTK

type C = CNTKLib
Layers.trace := true

//ensure latest nvidia driver is installed

//uncomment to set device global to CPU - defaults to GPU
//device <- DeviceDescriptor.CPUDevice  

(*
    Variational Autoencoder as a generative model
    (see basic autoencoder reference: https://cntk.ai/pythondocs/CNTK_105_Basic_Autoencoder_for_Dimensionality_Reduction.html)
    https://github.com/hwalsuklee/tensorflow-mnist-VAE/blob/master/mnist_data.py
*)

let isFast = false

let featureStreamName = "features" //need stream names later
let labelsStreamName = "labels"

let create_reader path is_training input_dim num_label_classes  = 
    let featureStream =  new StreamConfiguration(featureStreamName, input_dim)   
    let labelStream = new StreamConfiguration(labelsStreamName, num_label_classes)
    MinibatchSource.TextFormatMinibatchSource(
        path, 
        ResizeArray [featureStream; labelStream],
        (if is_training then MinibatchSource.InfinitelyRepeat else MinibatchSource.InfinitelyRepeat (*1uL*)),
        is_training)

//assume data is downloaded and extracted - see python tutorial
let cntk_samples_folder = @"c:\s\Repos\cntk\Examples\Image\DataSets\MNIST" //from CNTK download
let train_file = Path.Combine(cntk_samples_folder,"Train-28x28_cntk_text.txt" )
let test_file = Path.Combine(cntk_samples_folder, "Test-28x28_cntk_text.txt")

let input_dim = 784 //28*28
let encoding_dim = 512
let ouput_dim = input_dim
let num_label_classes = 10
let n_z = 2    // 2 dimensional latent space

//let init() = C.GlorotNormalInitializer()
let init() = C.HeNormalInitializer()

let input =  Node.Input(D input_dim, dynamicAxes=[Axis.DefaultBatchAxis()])

let recognition (features:Node) =
    let h_flat = L.Dense(D encoding_dim, activation=Activation.ELU, init=init()) features
    let w_mean = L.Dense(D n_z, name="w_mean", init=init()) h_flat
    let w_stddev = 1e-6 + (L.Dense(D n_z, name="w_stddev", init=init()) >> O.softplus) h_flat //stdv must be positive so softplus
    w_mean, w_stddev

let generation(z:Node) =
    let decode = 
        L.Dense(D encoding_dim, activation=Activation.ELU, init=init()) 
        >> L.Dense(D input_dim, activation=Activation.Sigmoid, init=init()) 
    decode z

let train_and_test (reader_train:MinibatchSource) (reader_test:MinibatchSource)  (encode_func:Node->Node*Node) (decode_func:Node->Node) = 
    let x = input ./ 255.  // rescaled 0 to 1

    let mu,sigma = encode_func x 
    let samples = O.random_normal(D n_z, 0., 1.) //zero mean, unit variance
    let guessed_z = mu + (sigma .* samples)
    let y = decode_func guessed_z

    let recnstr_loss_vctr = x .* O.log(y)  + (1.0 - x) .* O.log(1.0 - y)
    let recnstr_loss = O.reduce_sum(recnstr_loss_vctr,axis=new Axis(0))

    let kl_dvrgnc_vctr = O.square mu + O.square sigma - O.log(1e-8 + O.square(sigma)) - 1.0
    let kl_dvrgnc = 0.5 .* O.reduce_sum(kl_dvrgnc_vctr,axis=new Axis(0))

    let ELBO = O.reduce_mean(recnstr_loss,axis=0) - O.reduce_mean(kl_dvrgnc,axis=0)
    let loss = - ELBO

    loss.Func.Save(@"C:\s\repodata\fscntk\cntk_105b\fs_loss.bin")

    let epoch_size = 30000
    let minibatch_size = 64
    let num_sweeps_to_train_with = if isFast then 5 else 200
    let num_samples_per_sweep = 60000
    let num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
    
    let lr_schedule = T.schedule_per_sample (3e-3)

    let momentum_schedule = T.schedule(0.9126265014311797, minibatch_size)

    let opts = new AdditionalLearningOptions()
    opts.gradientClippingWithTruncation <- true  // mimic defaults used in python 
    //opts.gradientClippingThresholdPerSample <- 2.3

    //let learner = C.FSAdaGradLearner( 
    //                 O.parms y |> parmVector,
    //                 lr_schedule,
    //                 momentum_schedule,
    //                 true,
    //                 T.schedule_per_sample(0.9999986111120757), //variance momentum
    //                 opts
    //                 )

    let learner = C.AdamLearner( 
                     O.parms y |> parmVector,
                     lr_schedule,
                     momentum_schedule)
                     //true,
                     //T.schedule_per_sample(0.9999986111120757), //variance momentum
                     //1e-8
                     //)
                     
    //let learner = C.AdaDeltaLearner(
    //                 O.parms y |> parmVector,
    //                 lr_schedule)
                     //0.95,
                     //1e-8,
                     //opts
                     //)

    let trainer = Trainer.CreateTrainer(
                        null,
                        loss.Func,
                        loss.Func,
                        ResizeArray[learner]
                    )

    let featureStreamInfo = reader_train.StreamInfo(featureStreamName) 
    let labelStreamInfo = reader_train.StreamInfo(labelsStreamName)  
     
    let mutable aggregate_metric = 0.
    for i in 1 .. num_minibatches_to_train do
        let data = reader_train.GetNextMinibatch(uint32 minibatch_size)
        
        let input_map = 
            idict [ 
                input.Var, data.[featureStreamInfo] //same input / output of autoencoder
                //label.Var, data.[featureStreamInfo]
                ]
        let r = trainer.TrainMinibatch(input_map, device)
        let samples =  trainer.PreviousMinibatchSampleCount()
        let batchLoss = trainer.PreviousMinibatchEvaluationAverage() * float samples
        let lr = learner.LearningRate()
        printfn "batch %d loss:%f %d lr:%f" i batchLoss samples lr

        aggregate_metric <- aggregate_metric + batchLoss

    let train_error = aggregate_metric * 100. / float (trainer.TotalNumberOfSamplesSeen())
    printfn "Average training  error: %0.2f" train_error

    let test_minibatch_size = 32
    let num_samples = 10000
    let num_minibatches_to_test = num_samples / test_minibatch_size

    let mutable test_result = 0.
    let mutable metric_numer = 0.
    let mutable metric_denom = 0.

    for i in 1 .. num_minibatches_to_test do
        let data = reader_test.GetNextMinibatch(uint32 test_minibatch_size)

        let input_map = 
            let dt = new UnorderedMapVariableMinibatchData()
            dt.[input.Var] <- data.[featureStreamInfo]
            //dt.[label.Var] <- data.[featureStreamInfo]
            dt

        let eval_error = trainer.TestMinibatch(input_map, device)
        metric_numer <- metric_numer + abs(eval_error * float test_minibatch_size)
        metric_denom <- metric_denom + float test_minibatch_size

    let test_error = metric_numer * 100. / metric_denom
    printfn "Average test error: %0.2f" test_error

    y, train_error, test_error

let reader_train = create_reader train_file true (uint32 input_dim) (uint32 num_label_classes)

let reader_test = create_reader test_file false (uint32 input_dim) (uint32 num_label_classes)
;;
let model, simple_ae_train_error, simple_ae_test_error = train_and_test reader_train reader_test recognition generation
;;

let reader_eval = create_reader test_file false (uint32 input_dim) (uint32 num_label_classes)
let eval_minibatach_size = 50u
let eval_data = reader_eval.GetNextMinibatch(eval_minibatach_size)

let img_data = 
    eval_data.[reader_eval.StreamInfo(featureStreamName)].data 
    |> V.getArray 
    |> Array.head 
    |> Array.chunkBySize input_dim

let idx = Probability.RNG.Value.Next(int eval_minibatach_size)

let orig_image = img_data.[idx]
let decode_image = model |> E.eval1 (idict [input.Var, V.toValue(orig_image, D input_dim)]) |> Array.head |> Array.map ((*) 255.f)

let img = decode_image |> Array.map byte |> ImageUtils.toGray (28,28)
ImageUtils.show img




