#load "SetEnv.fsx"
open FsCNTK
open FsCNTK.FsBase
open FsCNTK.Layers
open Layers_Dense
open Layers_BN
open Layers_ConvolutionTranspose2D
open Layers_Convolution2D
open Layers_Recurrence
open CNTK
open System.IO
open FsCNTK.FsBase
open System

//Language Understanding with Recurrent Networks
//the code is based on the python example included with CNTK git or binary release
//under the Examples\LanguageUnderstanding\ATIS\Python folder
//
//See also this tutorial for background documentation: 
// https://cntk.ai/pythondocs/CNTK_202_Language_Understanding.html

//Note: The training results are not the same as for the Python version
//The Python model continues to reduce loss whereas this model
//stops reducing loss much earlier
//Ostensibly the two models are the same in terms of the
//number of parameters and tensors to train. 
//The computuational graph looks ok as well but why the difference - I don't know

type C = CNTKLib
Layers.trace := true

//Folder containing ATIS files which are
//part of the CNTK binary release download or CNTK git repo
let folder = @"D:\Repos\cntk\Examples\LanguageUnderstanding\ATIS\Data"

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
let create_model() =
  L.Embedding(D emb_dim, name="embed")
  >> L.Recurrence(L.LSTM(D hidden_dim, enable_self_stabilization=false), go_backwards=false, init_value=0.1)
  >> List.head
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
            dynamicAxes = [Axis.DefaultDynamicAxis(); Axis.DefaultBatchAxis()], //Sequence due to dynamic axis
            isSparse=true                                                       //SparseTensor
          )

//label sequence
let y = Node.Input
          (
            D num_labels, 
            dynamicAxes = [Axis.DefaultDynamicAxis(); Axis.DefaultBatchAxis()], 
            isSparse=true
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
let train (reader:MinibatchSource) model_func max_epochs task =

  let model = model_func x
  //let model = Function.Load(@"D:\repodata\fscntk\l_py_m.bin",device) |> F
  //let xVar = model.Func.Arguments.[0]
 

  let loss,label_error = create_criterion_function model y

  model.Func.Save(@"D:\repodata\fscntk\l_fs_m.bin")
  loss.Func.Save(@"D:\repodata\fscntk\l_fs_l.bin")

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

  //let momentums = schedule [1,0.9048374180359595] minibatch_size 
  //below seems be the correct translation of the python code
  let m = new TrainingParameterScheduleDouble(0.9048374180359595,uint32 minibatch_size)
  //let m = C.MomentumAsTimeConstantSchedule(new DoubleVector(ResizeArray[0.9048374180359595]),uint32 max_epochs)
  //let m = C.MomentumAsTimeConstantSchedule(momentums)
  //let m = C.MomentumAsTimeConstantSchedule(0.9048374180359595)


  let options = new AdditionalLearningOptions()
  options.gradientClippingThresholdPerSample <- 15.0
  options.gradientClippingWithTruncation <- true
  let learner = C.AdamLearner(
                      O.parms model |> parmVector ,
                      lr_schedule,
                      m,
                      true,                                   //should be exposed by CNTK C# API as C.DefaultUnitGainValue()
                      constSchedule 0.9999986111120757,       //from python code
                      1e-8,                                   //from python code 
                      false,                                  //from python code
                      options)

  let learner = C.AdamLearner(
                  O.parms model |> parmVector,
                  lr_schedule,
                  m)

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
    | Slot_Tagging -> idict [x.Var,data.[query]; y.Var,data.[labels]]
    | Intent       -> idict [x.Var,data.[query]; y.Var,data.[intent]]

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
  let model_func = create_model()

  let reader = create_reader (Path.Combine(folder,"atis.train.ctf")) true
  let task = current_task
  let model = train reader model_func 10 task
  model.Func.Save(modelFile task)


//evaluate model for the given task
//the model is read from the trained model 
//saved to file
let evaluate (reader:MinibatchSource)  task =
  let model = Function.Load(modelFile task, device)
  let loss,label_error = create_criterion_function (F model) y

  let labels = reader.StreamInfo(sLabels)
  let query  = reader.StreamInfo(sQuery)
  let intent = reader.StreamInfo(sIntent)
  
  let queryVar = loss.Func.Arguments.[0]
  let labelsVar = loss.Func.Arguments.[1]

  let eval = C.CreateEvaluator(loss.Func)

  let mutable go = true
  while go do
    let minibatch_size = 500
    let data = reader.GetNextMinibatch(uint32 minibatch_size) //get minibatch
    if data.Count = 0 then
      go <- false
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

(* comparison with python model
let model = Function.Load(modelFile current_task, device)
let pyModel = Function.Load(@"D:\repodata\fscntk\l_py_m.bin",device)
let f x = x |> Seq.filter (fun (p:Parameter)->p.Uid.StartsWith("Input",StringComparison.CurrentCultureIgnoreCase)) |> Seq.toArray
let b1  = model.Parameters() |> f
let b2  = pyModel.Parameters() |> f
Seq.zip (model.Parameters()) (pyModel.Parameters()) 
|> Seq.toArray 
|> Array.map(fun (a,b)->a.Name,a.Uid,!++ a.Shape,"##", b.Name,b.Uid, !++ b.Shape)

//ordering is different among parameters between F# and python models - does is matter?
val it : (string * string * Shape * string * string * string * Shape) [] =
  [|("W", "Parameter109", Ds [300; 129], "##", "W", "Parameter11549",
     Ds [300; 129]);
    ("b", "Parameter10", Ds [1200], "##", "b", "Parameter11550", Ds [129]);
    ("W", "Parameter11", Ds [150; 1200], "##", "b", "Parameter11100",
     Ds [1200]);
    ("embed", "Parameter2", Ds [943; 150], "##", "W", "Parameter11101",
     Ds [150; 1200]);
    ("H", "Parameter12", Ds [300; 1200], "##", "H", "Parameter11102",
     Ds [300; 1200]);
    ("b", "Parameter110", Ds [129], "##", "E", "Parameter11089", Ds [943; 150])|]
*)

//actually test the model
let do_test() =
  let reader = create_reader (Path.Combine(folder,"atis.test.ctf")) false
  evaluate reader current_task


(*
do_train()


do_test() 
*)

(*
let z1 = create_model() x
z1.Func.Save(@"D:\repodata\fscntk\m_fs_untrained.bin")

*)

(*
save and reload to check model and loss

let z = create_model() x
let l,a = create_criterion_function z y
let mf = @"D:\repodata\fscntk\l_fs_m.bin"
let lf = @"D:\repodata\fscntk\l_fs_l.bin"
z.Func.Save(mf)
l.Func.Save(lf)

let z' = Function.Load(mf,device)
let l' = Function.Load(lf,device)
*)
