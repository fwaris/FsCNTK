#load "SetEnv.fsx"
open FsCNTK
open FsCNTK.FsBase
open FsCNTK.Layers
open Layers_Dense
open Layers_BN
open Layers_ConvolutionTranspose2D
open Layers_Convolution2D
open Layers_Sequence
open CNTK
open System.IO
open FsCNTK.FsBase
open System

let getValue (v:Variable) = 
  let vv = v.GetValue()
  let vvv = new Value(vv)
  let vvvv = vvv.GetDenseData<float32>(v)
  printfn "%A" v.Shape.Dimensions
  printfn "%A" (vvvv |> Seq.map Seq.average |> Seq.average)
  vvvv |> Seq.collect (fun x->x)

(*
Language Understanding with Recurrent Networks
the code is based on the python example included with CNTK git or binary release
under the Examples\LanguageUnderstanding\ATIS\Python folder

See also this tutorial for background documentation: 
 https://cntk.ai/pythondocs/CNTK_202_Language_Understanding.html

Note: Visualization of the LSTM model is in the Scripts/imgs folder

*)
type C = CNTKLib
Layers.trace := true

//Folder containing ATIS files which are
//part of the CNTK binary release download or CNTK git repo
let folder = @"c:\s\Examples\LanguageUnderstanding\ATIS\Data"

let vocab_size = 943 
let num_labels = 129
let num_intents = 26

//model dimensions
let input_dim  = vocab_size
let label_dim  = num_labels
let emb_dim    = 150
let hidden_dim = 300

//input data streams 
let sQuery,sIntent,sLabels = "S0","S1","S2"
let streamConfigurations = 
    ResizeArray<StreamConfiguration>(
        [
            new StreamConfiguration(sQuery, vocab_size, true)    
            new StreamConfiguration(sIntent, num_intents,true)
            new StreamConfiguration(sLabels, num_labels,true)
        ]
        )

//reader
let create_reader path is_training =
  MinibatchSource.TextFormatMinibatchSource
    (
      path,
      streamConfigurations,
      (if is_training then  MinibatchSource.InfinitelyRepeat else MinibatchSource.FullDataSweep),
      is_training
    )

//model
let cVal = new Constant(!> [| NDShape.InferredDimension |], dataType, 0.1) :> Variable |> V 

let create_model() =
  let cell = L.LSTM(D hidden_dim,enable_self_stabilization=false)
  L.Embedding(D emb_dim, name="embed")
  >> L.Recurrence(initial_states=[cVal;cVal], go_backwards=false) cell
  >> O.getOutput 0
  >> L.Dense(D num_labels, name="classify")

//loss function
let create_criterion_function z y =
  let ce = O.cross_entropy_with_softmax(z, y)
  let err = O.classification_error(z, y)
  ce,err

//query sequence
let x = Node.Input
          (
            D vocab_size, 
            dynamicAxes = [Axis.DefaultDynamicAxis(); Axis.DefaultBatchAxis()] //Sequence due to dynamic axis
          )

//label sequence
let y = Node.Input
          (
            D num_labels, 
            dynamicAxes = [Axis.DefaultDynamicAxis(); Axis.DefaultBatchAxis()] 
          )

//training tasks 
type Task = Slot_Tagging | Intent

//set which task to use for this run
let current_task = Slot_Tagging

//model file name function for saving model based on task selected
let modelFile = function
  | Slot_Tagging -> "slot_model.bin" 
  | Intent -> "intent_model.bin"

//train given model
let train (reader:MinibatchSource) (model:Node) max_epochs task =

  let xVar = model.Func.Arguments.[0]

  let loss,label_error = create_criterion_function model y
  //loss.Func.Save(@"D:\repodata\fscntk\l_fs2_l.bin")

  let parms = O.parms model 
  let totalParms = parms |> Seq.map (fun p -> p.Shape.TotalSize) |> Seq.sum
  printfn "Training %d parameters in %d parameter tensors" totalParms (Seq.length parms)

  //serializing - deserializing model validates the model structure better
  //model.Func.Save(Path.Combine(@"D:\repodata\fscntk","TestLstm_model.bin"))
  //let g = Function.Load(@"D:\repodata\fscntk\TestLstm_model.bin",device)
  
  let epoch_size = 18000
  let minibatch_size = 70

  let lr_per_sample = [for _ in 0..4 -> 1,3e-4] @ [1,1.5e-4]
  let lr_per_minibatch = lr_per_sample |> List.map (fun (i,r) -> i, r * float minibatch_size)
  let lr_schedule = schedule lr_per_minibatch epoch_size

  let momentum = new TrainingParameterScheduleDouble(0.9048374180359595,uint32 minibatch_size)

  let options = new AdditionalLearningOptions()
  options.gradientClippingThresholdPerSample <- 15.0
  options.gradientClippingWithTruncation <- true
  let learner = C.AdamLearner(
                      O.parms model |> parmVector ,
                      lr_schedule,
                      momentum,
                      true,                                   //should be exposed by CNTK C# API as C.DefaultUnitGainValue()
                      constSchedule 0.9999986111120757,       //from python code
                      1e-8,                                   //from python code 
                      false,                                  //from python code
                      options)

  let trainer = C.CreateTrainer(
                      model.Func,
                      loss.Func, 
                      label_error.Func,
                      lrnVector [learner])

  let labels = reader.StreamInfo(sLabels)
  let query  = reader.StreamInfo(sQuery)
  let intent = reader.StreamInfo(sIntent)
  
  let data_map (data:UnorderedMapStreamInformationMinibatchData) = 
    match task with
    | Slot_Tagging -> idict [xVar,data.[query]; y.Var,data.[labels]]
    | Intent       -> idict [xVar,data.[query]; y.Var,data.[intent]]

  let mutable t = 0
  for epoch in 1..max_epochs do         // loop over epochs
    let mutable avgLoss = 0.0
    let mutable avgAcc = 0.0
    let mutable epochSamples = 0.0
    let epoch_end = epoch * epoch_size
    while t < epoch_end do                                        // loop over minibatches on the epoch
        let data = reader.GetNextMinibatch(uint32 minibatch_size, device) //get minibatch
        let r = trainer.TrainMinibatch(data_map data, device)     // update model with it
        let mbSamples = float data.[query].numberOfSamples
        t <- t + int mbSamples                                      // samples so far
        epochSamples <- epochSamples + mbSamples
        let mbLoss = trainer.PreviousMinibatchLossAverage()
        let acc = System.Math.Round(trainer.PreviousMinibatchEvaluationAverage() * 100.0, 2)
        avgLoss <- avgLoss + (mbLoss * mbSamples)
        avgAcc <- avgAcc + (acc * mbSamples)

    let samplesSeen = trainer.PreviousMinibatchSampleCount()
    let acc = System.Math.Round(trainer.PreviousMinibatchEvaluationAverage() * 100.0, 2)
    let lr = learner.LearningRate()
    let loss = avgLoss / epochSamples
    let acc = avgAcc / epochSamples
    printfn "Epoch: %d, Loss=%f * %d, metric=%f, Total Samples=%d, LR=%f" epoch loss (int epochSamples) acc t lr
    

    trainer.SummarizeTrainingProgress() //does not work due to that fact that
                                        //its not possible to specify progress writers in .Net 
                                        //probably because a progress writer needs to inherit from a C++ class
                                        //inheriting from a .Net SWIG proxy is not the same
  model //return trained model
    

//actually does the training for the selected task
let do_train() =
  let model_func = create_model() x

  let reader = create_reader (Path.Combine(folder,"atis.train.ctf")) true
  let task = current_task
  let model = train reader model_func 10 task
  model.Func.Save(modelFile task)


//evaluate model for the given task
//the model is read from the trained model 
//saved to file, earlier
let evaluate (reader:MinibatchSource)  task =
  let model = Function.Load(modelFile task, device)
  let loss,label_error = create_criterion_function (F model) y

  let labels = reader.StreamInfo(sLabels)
  let query  = reader.StreamInfo(sQuery)
  let intent = reader.StreamInfo(sIntent)
  
  let queryVar = loss.Func.Arguments.[0]
  let labelsVar = loss.Func.Arguments.[1]

  let eval = C.CreateEvaluator(loss.Func)

  let rec run (lossAcc, samplesAcc) =
    let minibatch_size = 500
    let data = reader.GetNextMinibatch(uint32 minibatch_size) //get minibatch
    if data.Count = 0 then
      lossAcc,samplesAcc
    else
      let samplesX = data.[query].numberOfSamples
      printfn "Evaluating %d sequences" samplesX
      let xV = data.[query].data
      let yV = data.[match task with Slot_Tagging -> labels | Intent -> intent].data
      let inputs = new UnorderedMapVariableMinibatchData()
      inputs.Add(queryVar,new MinibatchData(xV))
      inputs.Add(labelsVar,new MinibatchData(yV))
      let r = eval.TestMinibatch(inputs,device)
      printfn "%A" r
      run (r::lossAcc, samplesX::samplesAcc)

  let ls,smpls = run ([],[])

  printfn "metric %0.2f * %d" (List.average ls * 100.0) (List.sum smpls)


//actually test the model
let do_test() =
  let reader = create_reader (Path.Combine(folder,"atis.test.ctf")) false
  evaluate reader current_task


//use the trained model to predict the tag for each
//word in a query
let test_slot_tagging() =
  let queryFile = @"D:\Repos\cntk\Examples\LanguageUnderstanding\ATIS\BrainScript\query.wl"
  let slotsFile = @"D:\Repos\cntk\Examples\LanguageUnderstanding\ATIS\BrainScript\slots.wl"
  let query_dict = queryFile |> File.ReadLines |> Seq.mapi (fun i q -> q,i) |> Map.ofSeq
  let slots_wl = slotsFile |> File.ReadLines |> Seq.toArray
  let slot_dict  = slots_wl |> Seq.mapi (fun i q -> q,i) |> Map.ofSeq

  let query = "BOS flights from new york to seattle EOS"
  let words = query.Split([|' '|])

  let ws =  words |> Array.map (fun w -> query_dict.[w])
  
  printfn "Encoded words %A" (Array.zip words ws)
  let zeros() = Array.create query_dict.Count 0.f
  let onehot = ws |> Array.map (fun w->let a = zeros() in a.[w] <- 1.f; a)
  //slot_dict.Count
  //query_dict.["BOS"]
  //onehot.[0] |> Array.findIndex (fun x->x = 1.0f)
  //onehot.[0].[178]

  let model = Function.Load(modelFile Slot_Tagging, device)   //load model from temp file created during training 
  let xVar = model.Arguments.[0]
  let yVar = model.Outputs.[0]
  let v = Value.CreateSequence(xVar.Shape, onehot |> Array.collect yourself, device)
  let outp = idict[yVar,(null:Value)]
  model.Evaluate(idict[xVar,v],outp,device)

  let ys = outp.[yVar].GetDenseData<float32>(yVar)  

  let best = 
    ys.[0] 
    |> Seq.chunkBySize yVar.Shape.Dimensions.[0] 
    |> Seq.map (fun xs-> 
      xs 
      |> Seq.mapi (fun i x->i,x) 
      |> Seq.maxBy snd |> fst) //take the index of the max value
    |> Seq.toArray
  printfn "Best: %A" best

  let tagged =
    Array.zip
      words
      (best |> Array.map (fun i->slots_wl.[i]))
  printfn "Tagged: %A" tagged

//

(*
do_train()

do_test() 

test_slot_tagging()

*)
