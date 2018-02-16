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

//Language Understanding with Recurrent Networks
//the code is based on the python example included with CNTK git or binary release
//under the Examples\LanguageUnderstanding\ATIS\Python folder
//
//See also this tutorial for background documentation: 
// https://cntk.ai/pythondocs/CNTK_202_Language_Understanding.html

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


//reader 
let sQuery,sIntent,sLabels = "S0","S1","S2"
let streamConfigurations = 
    ResizeArray<StreamConfiguration>(
        [
            new StreamConfiguration(sQuery, vocab_size, true)    
            new StreamConfiguration(sIntent, num_intents,true)
            new StreamConfiguration(sLabels,num_labels,true)
        ]
        )

let create_reader path is_training =
  MinibatchSource.TextFormatMinibatchSource
    (
      path,
      streamConfigurations,
      (if is_training then  MinibatchSource.InfinitelyRepeat else 1UL),
      is_training
    )

//model
let create_model() =
  L.Embedding(D emb_dim,name="embed")
  >> L.Recurrence(L.LSTM(D hidden_dim),go_backwards=false)
  >> List.head
  >> L.Dense(D num_labels, name="classify")

let create_criterion_function z y =
  let ce = O.cross_entropy_with_softmax(z, y)
  let err = O.classification_error(z, y)
  ce,err

let x = Node.Variable
          (
            D vocab_size, 
            dynamicAxes = [Axis.DefaultDynamicAxis(); Axis.DefaultBatchAxis()], //Sequence due to dynamic axis
            isSparse=true                                                       //SparseTensor
          )

let y = Node.Variable
          (
            D num_labels, 
            dynamicAxes = [Axis.DefaultDynamicAxis(); Axis.DefaultBatchAxis()], 
            isSparse=true
          )

type Task = Slot_Tagging | Intent

let train (reader:MinibatchSource) model_func max_epochs task =

  let model = model_func x
  let loss,label_error = create_criterion_function model y

  model.Func.Save(Path.Combine(@"D:\repodata\fscntk","TestLstm_model.bin"))
  let g = Function.Load(@"D:\repodata\fscntk\TestLstm_model.bin",device)
  
  let epoch_size = 18000
  let minibatch_size = 70

  let lr_per_sample = [for _ in 1..4 -> 1,3e-4] @ [1,1.5e-4]
  let lr_per_minibatch = lr_per_sample |> List.map (fun (i,r) -> i, r * float epoch_size)
  let lr_schedule = schedule lr_per_minibatch epoch_size

  let momentums = schedule [1,0.9048374180359595] minibatch_size 

  let learner = C.AdamLearner(
                      O.parms model |> parmVector ,
                      lr_schedule,
                      momentums)

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
    let epoch_end = (epoch+1) * epoch_size
    while t < epoch_end do              // loop over minibatches on the epoch
        let data = reader.GetNextMinibatch(uint32 minibatch_size) //get minibatch
        let r = trainer.TrainMinibatch(data_map data, device)        // update model with it
        t <- t + data.Count                   // samples so far

        if t % 100 = 0 then
          let mbLoss = trainer.PreviousMinibatchLossAverage()
          printfn "Minibatch: %d, Loss=%f" t mbLoss
    
    trainer.SummarizeTrainingProgress()
    
let do_train() =
  let z = create_model()
  let m = z x
  let reader = create_reader (Path.Combine(folder,"atis.train.ctf")) true
  train reader z 10 Slot_Tagging

(*
do_train()
*)





