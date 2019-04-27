#load "SetEnv.fsx"
open FsCNTK
open CNTK
open System.IO

type C = CNTKLib
Layers.trace := true

//ensure latest nvidia driver is installed

//uncomment to set device global to CPU - defaults to GPU
//device <- DeviceDescriptor.CPUDevice  

(*
    Autoencoder for dimensionality reduction
    Based on: https://cntk.ai/pythondocs/CNTK_105_Basic_Autoencoder_for_Dimensionality_Reduction.html
*)

let isFast = true

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
let encoding_dim = 32
let ouput_dim = input_dim
let num_label_classes = 10

let create_model (features:Node) =
    let init() = C.GlorotNormalInitializer()
    let scaled_features = features ./ 255.
    let encode = L.Dense(D encoding_dim, activation=Activation.ReLU, init=init()) scaled_features
    let decode = L.Dense(D input_dim, activation=Activation.Sigmoid, init=init()) encode
    decode

let input =  Node.Input(D input_dim, dynamicAxes=[Axis.DefaultBatchAxis()])
let label = Node.Input(D input_dim, dynamicAxes=[Axis.DefaultBatchAxis()])

let train_and_test (reader_train:MinibatchSource) (reader_test:MinibatchSource)  model_func = 

    let model:Node = model_func input

    let target = label ./ 255.
    let loss = - (target .* O.log(model)) + ((1.0 - target) .* O.log(1.0 - model))
    let label_error = O.classification_error(model,target)
    loss.Func.Save(@"C:\s\repodata\fscntk\cntk_105\fs_loss.bin")

    let epoch_size = 30000
    let minibatch_size = 64
    let num_sweeps_to_train_with = if isFast then 5 else 100
    let num_samples_per_sweep = 60000
    let num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
    
    let lr_schedule = T.schedule_per_sample (3e-5)

    let momentum_schedule = T.schedule(0.9126265014311797, minibatch_size)

    let opts = new AdditionalLearningOptions()
    opts.gradientClippingWithTruncation <- true  // mimic defaults used in python 

    let learner = C.FSAdaGradLearner( 
                     O.parms model |> parmVector,
                     lr_schedule,
                     momentum_schedule,
                     true,
                     T.schedule_per_sample(0.9999986111120757), //variance momentum
                     opts
                     )

    let trainer = Trainer.CreateTrainer(
                        model.Func,
                        loss.Func,
                        label_error.Func,
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
                label.Var, data.[featureStreamInfo]
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
            dt.[label.Var] <- data.[featureStreamInfo]
            dt

        let eval_error = trainer.TestMinibatch(input_map, device)
        metric_numer <- metric_numer + abs(eval_error * float test_minibatch_size)
        metric_denom <- metric_denom + float test_minibatch_size

    let test_error = metric_numer * 100. / metric_denom
    printfn "Average test error: %0.2f" test_error

    model, train_error, test_error

let reader_train = create_reader train_file true (uint32 input_dim) (uint32 num_label_classes)

let reader_test = create_reader test_file false (uint32 input_dim) (uint32 num_label_classes)
;;
let model, simple_ae_train_error, simple_ae_test_error = train_and_test reader_train reader_test create_model

;;
let reader_eval = create_reader test_file false (uint32 input_dim) (uint32 num_label_classes)
let eval_minibatach_size = 50u
let eval_data = reader_eval.GetNextMinibatch(eval_minibatach_size)

let img_data = 
    eval_data.[reader_eval.StreamInfo(featureStreamName)].data 
    |> Vl.getArray 
    |> Array.chunkBySize input_dim


let idx = Probability.RNG.Value.Next(int eval_minibatach_size)

let orig_image = img_data.[idx]
let decode_image = model |> E.eval1 (idict [input.Var, Vl.toValue(orig_image, D input_dim)]) |> Array.map ((*) 255.f)

let img = decode_image |> Array.map byte |> ImageUtils.toGray (28,28)
ImageUtils.show img



