#load "FsCNTK_SetEnv.fsx"
open FsCNTK
open CNTK
open System.IO

type C = CNTKLib
Layers.trace := true

//ensure latest nvidia driver is installed

//uncomment to set device global to CPU - defaults to GPU
//device <- DeviceDescriptor.CPUDevice  

(*
    Factor VAE - a generative model driven by a disentagled latent representation
    "Distanging by Factorizing", Kim, et. al https://arxiv.org/pdf/1802.05983.pdf

    (see basic autoencoder reference: https://cntk.ai/pythondocs/CNTK_105_Basic_Autoencoder_for_Dimensionality_Reduction.html)
    https://github.com/hwalsuklee/tensorflow-mnist-VAE/blob/master/mnist_data.py

    )
    
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
let n_z = 10
let disc_dim = 1000
let gamma = 100.0

//let init() = C.GlorotNormalInitializer()
let init() = C.HeNormalInitializer()

let input =  Node.Input(D input_dim, dynamicAxes=[Axis.DefaultBatchAxis()])

let discriminator()  =
    let logits = 
        let layers = seq {for i in 1 .. 5 -> L.Dense(D disc_dim, activation=Activation.LeakyReLU(Some 0.2), init=init())}
        let dense = (Seq.head layers, Seq.tail layers) ||> Seq.fold ( >> )
        dense>>L.Dense(D 2, init=init())
    let prob = fun n -> O.softmax (logits n,axis=new Axis(0))
    (fun (x:Node) -> logits x), //return two functions for logit and prob 
    (fun (x:Node) -> prob x)

let recognition (features:Node) =
    let h_flat = (B.Stabilizer()>>L.Dense(D encoding_dim, activation=Activation.ELU, init=init())) features
    let w_mean = L.Dense(D n_z, name="w_mean", init=init()) h_flat
    let w_stddev_logits = L.Dense(D n_z, name="w_stddev", init=init()) h_flat
    w_mean, w_stddev_logits

let generation =
    let decode = 
        L.Dense(D encoding_dim, activation=Activation.ELU, init=init())              //declare generator model statically 
        >> L.Dense(D input_dim, activation=Activation.Sigmoid, init=init())          //the parameters are allocated here 
                                                                                     //they are updated at training
                                                                                     //this allows us to use generator separately later
    fun (z:Node) -> decode z                                                         //return a function so that the weights can be re-used with different input variables

let train_and_test (reader_train:MinibatchSource) (reader_test:MinibatchSource)  (encode_func:Node->Node*Node) (decode_func:Node->Node) = 
    let x = input ./ 255.  // rescaled 0 to 1
    let minibatch_size = 64

    let Dsc,Dsc_prob = discriminator() //allocate parameters

    let mu,sigma_logits = encode_func x 
    let sigma = O.exp(sigma_logits ./ 2.)
    let samples = O.random_normal(D n_z, 0., 1.) //zero mean, unit variance
    let guessed_z = mu + (sigma .* samples)      //reparametrize trick
    let recnstr_x = decode_func guessed_z

    let perm_z = O.randperm(guessed_z, minibatch_size)

    let D_z_logits = Dsc guessed_z

    let D_z_prob      = Dsc_prob guessed_z
    let D_z_perm_prob = Dsc_prob perm_z

    //this matches the paper and https://github.com/paruby/FactorVAE/blob/master/factor_vae.py
    let D_tc_loss2 = - 0.5 .* (O.reduce_mean(O.log(1e-8 + D_z_prob.[0..1]),new Axis(0)) + 
                              O.reduce_mean(O.log(1e-8 + D_z_perm_prob.[1..2]),new Axis(0)))

    //from: https://github.com/paruby/FactorVAE/blob/master/factor_vae.py
    //# FactorVAE paper has gamma * log(D(z) / (1- D(z))) in Algorithm 2, where D(z) is probability of being real
    //# Let PT be probability of being true, PF be probability of being false. Then we want log(PT/PF)
    //# Since PT = exp(logit_T) / [exp(logit_T) + exp(logit_F)]
    //# and  PT = exp(logit_F) / [exp(logit_T) + exp(logit_F)], we have that
    //# log(PT/PF) = logit_T - logit_F
    let vae_tc_loss = gamma .* O.reduce_mean( D_z_logits.[0 .. 1]  - D_z_logits.[1 .. 2]  , new Axis(0))

    let recnstr_loss_vctr = (x .* O.log(recnstr_x))  + ((1.0 - x) .* O.log(1.0 - recnstr_x))
    let recnstr_loss = - O.reduce_sum(recnstr_loss_vctr, new Axis(0))

    let variance = O.exp(sigma_logits)
    let kl_dvrgnc_vctr = (1.0 + sigma_logits - O.square mu - variance)
    let kl_dvrgnc = -0.5 .* O.reduce_mean(kl_dvrgnc_vctr,new Axis(0))

    let loss = recnstr_loss + kl_dvrgnc + vae_tc_loss 

    loss.Func.Save(@"C:\s\repodata\fscntk\cntk_105b\fs_loss.bin")
     
    let epoch_size = 30000
    let num_sweeps_to_train_with = if isFast then 10 else 600
    let num_samples_per_sweep = 60000
    let num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
    
    let lr_schedule = T.schedule_per_sample (3e-4)
    let lr_schedule2 = T.schedule_per_sample (3e-4)

    let momentum_schedule = T.schedule(0.9126265014311797, minibatch_size)

    let opts = new AdditionalLearningOptions()
    opts.gradientClippingWithTruncation <- true  // mimic defaults used in python 
    opts.gradientClippingThresholdPerSample <- 2.3

    let vae_learner = C.AdamLearner( 
                         O.parms recnstr_x |> parmVector,
                         lr_schedule,
                         momentum_schedule,
                         true,
                         T.momentum_schedule 0.9999986111120757,
                         1e-8,
                         false,
                         opts)

    let vae_trainer = Trainer.CreateTrainer(
                        null,
                        loss.Func,
                        loss.Func,
                        ResizeArray[vae_learner]
                    )

    let dsc_learner = C.AdamLearner( 
                         O.parms D_z_logits |> parmVector,
                         lr_schedule2,
                         momentum_schedule,
                         true,
                         T.momentum_schedule 0.9999986111120757,
                         1e-8,
                         false,
                         opts)


    let dsc_trainer = Trainer.CreateTrainer(
                        null,
                        D_tc_loss2.Func,
                        D_tc_loss2.Func,
                        ResizeArray[dsc_learner]
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
        let vae_r = vae_trainer.TrainMinibatch(input_map, device)
        let dsc_r = dsc_trainer.TrainMinibatch(input_map,device)

        let samples_vae =  vae_trainer.PreviousMinibatchSampleCount()
        let batchLoss_vae = vae_trainer.PreviousMinibatchEvaluationAverage() * float samples_vae

        let samples_dsc =  dsc_trainer.PreviousMinibatchSampleCount()
        let batchLoss_dsc = dsc_trainer.PreviousMinibatchEvaluationAverage() * float samples_dsc

        printfn "batch %d loss_vae:%f  loss_dsc:%f" i batchLoss_vae batchLoss_dsc

        aggregate_metric <- aggregate_metric + batchLoss_vae



    let train_error = aggregate_metric * 100. / float (dsc_trainer.TotalNumberOfSamplesSeen())
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

        let eval_error = vae_trainer.TestMinibatch(input_map, device)
        metric_numer <- metric_numer + abs(eval_error * float test_minibatch_size)
        metric_denom <- metric_denom + float test_minibatch_size

    let test_error = metric_numer * 100. / metric_denom
    printfn "Average test error: %0.2f" test_error

    recnstr_x, train_error, test_error

let reader_train = create_reader train_file true (uint32 input_dim) (uint32 num_label_classes)

let reader_test = create_reader test_file false (uint32 input_dim) (uint32 num_label_classes)
;;
let model, simple_ae_train_error, simple_ae_test_error = train_and_test reader_train reader_test recognition generation
;;

(* use validation set to reproduce images
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
*)

(*
*)

let zInput = Node.Input(D n_z, dynamicAxes=[Axis.DefaultBatchAxis()])
let zModel = generation zInput
let range = [-2.0; -1.0; 0.0; 1.0; 2.0]
let manifold =                      //vary only 1 dimension while keeping other fixed in batches of k
    [for i in 0 .. n_z - 1 do
        for r in range do
            for k in [-1.5; -1.0; 1.0; 1.5] do
                let ks = [| for i in 0 .. n_z - 1 -> k|]
                ks.[i] <- r
                yield ((i,k), ks)]

let byslice = 
    manifold
    |> List.groupBy (fst>>fst)
    |> List.map (fun (i,vs)-> i, vs |> List.sortBy (fun ((_,k),_) -> k) |> Seq.map snd |> Seq.toList)
    |> Map.ofList

byslice
    |> Map.map(fun i vs ->
        vs |> List.map(fun v ->
            let img = 
                zModel 
                    |> E.eval1 (idict [zInput.Var, Vl.toValue(v, D n_z)]) 
                    |> Array.map ((*) 255.f)
                    |> Array.map byte
                    |> ImageUtils.toGray (28,28)
            img))
    |> Map.iter (fun i imgs ->     
        let sz =  (imgs.Length |> float |> sqrt |> ceil |> int)
        ImageUtils.showGrid (string i) (sz,sz) imgs)


