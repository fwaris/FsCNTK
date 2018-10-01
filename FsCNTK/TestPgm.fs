module Pgm
//#load "..\Scripts\SetEnv.fsx"
//this file is only used for native debugging with 
//#load "..\Scripts\SetEnv.fsx"
//#r @"F:\Projects\Prognostics\packages\FSharp.Data.2.4.6\lib\net45\FSharp.Data.dll"
//#r @"D:\Repos\Packages\Packages\packages\FSharp.Data.2.4.6\lib\net45\FSharp.Data.dll"
open FsCNTK
open FsCNTK.FsBase
open Layers_Dense
open Layers_Dropout
open Layers_Sequence
open CNTK
open Probability
open FSharp.Data
open ValueInterop
open System
open System
//device <- DeviceDescriptor.CPUDevice
type CNTKLib = C
let dmap = float32
//C.SetCheckedMode(true)


let id_col = "unit_number"
let time_col = "time"
let feature_cols = ["op_setting_1"; "op_setting_2"; "op_setting_3"] @ [for i in 1..21 -> sprintf "sensor_measurement_%d" i]
let column_names = [id_col; time_col] @ feature_cols

(*
let train_orig = CsvFile.Load(@"https://raw.githubusercontent.com/daynebatten/keras-wtte-rnn/master/train.csv",hasHeaders=false)
train_orig.Save(@"D:\repodata\fscntk\wtte\train_org.csv")
let test_x_orig = CsvFile.Load(@"https://raw.githubusercontent.com/daynebatten/keras-wtte-rnn/master/test_x.csv",hasHeaders=false)
test_x_orig.Save(@"D:\repodata\fscntk\wtte\test_x_orig.csv")
let test_y_orig = CsvFile.Load(@"https://raw.githubusercontent.com/daynebatten/keras-wtte-rnn/master/test_y.csv",hasHeaders=false)
test_y_orig.Save(@"D:\repodata\fscntk\wtte\test_y_orig.csv")
*)

let train_orig = CsvFile.Load(@"D:\repodata\fscntk\wtte\train_org.csv", hasHeaders=false).Rows |> Seq.toArray
let test_x_orig = CsvFile.Load(@"D:\repodata\fscntk\wtte\test_x_orig.csv", hasHeaders=false).Rows |> Seq.toArray
let test_y_orig = CsvFile.Load(@"D:\repodata\fscntk\wtte\test_y_orig.csv", hasHeaders=false).Rows |> Seq.toArray

//scale features to -1,1 drop constant features
//(plan use ML.Net transforms and pipelines in future)
let all_data_scaled =
  let all_data = Array.append train_orig test_x_orig                //combine train/test before scaling
  let colIdx = column_names |> List.mapi(fun i c->c,i) |> Map.ofSeq //index column names
  let colMinMax =                                                   //min max of each column
    feature_cols 
    |> List.map(fun c -> 
      let idx = colIdx.[c]
      let cRows = all_data |> Seq.map (fun r -> r.[idx].AsFloat()) 
      c,(Seq.min cRows, Seq.max cRows))
    |> Map.ofList
  all_data                                                        //scale features (i.e. exclude id and time)
  |> Seq.map (fun r->
      seq {
       yield r.[colIdx.[id_col]].AsFloat()
       yield r.[colIdx.[time_col]].AsFloat()
       for f in feature_cols do
          let idx = colIdx.[f]
          let mn,mx = colMinMax.[f]
          if mx <> mn then                                //remove constants
            let v1 = r.[idx].AsFloat()
            let v2 = scaler (-1.0,1.0) colMinMax.[f]  v1  //scale to -1,1
            yield v2
      }
      |> Seq.map dmap
      |> Seq.toArray)
  |> Seq.toArray

printfn "rows:%d; cols:%d" all_data_scaled.Length all_data_scaled.[0].Length

all_data_scaled |> Array.countBy (fun r->r.[0])

//split back into train/test after scaling
let train = all_data_scaled.[0..train_orig.Length-1] 
let test = all_data_scaled.[train_orig.Length ..]

let max_days = 100-1


//function to get upto max-days history for an engine-day
let maxDayHist_X i (engineDays:'a[][]) (curDay:'a[]) = 
  let hist = 
    engineDays 
    |> Array.map (fun xs->xs.[2..])     //keep features
    |> Array.skip (max 0 (i-max_days))  //skip to just before the first history row
    |> Array.truncate (min i max_days)  //the # of rows of history to take (no padding needed in cntk)
  Array.append hist [| curDay.[2..] |]  //append current row to history

//returns float[][][] - [engine_day] [upto 100 days hist] [17 var feature vector]
let batchSeqHistX isTrain (data:'a[][])  = 
  data 
  |> Array.groupBy (fun xs->xs.[0]) //group by engine
  |> Array.collect(fun (engn,xs) ->
    if isTrain then                           //for training keep max-days history for each day, on a rolling basis
      xs |> Array.mapi (fun i ys -> maxDayHist_X i xs ys)
    else
      let ys = xs |> Array.last               //for test data just use max-day day history for the *last* day
      let i = xs.Length-1
      [| maxDayHist_X i xs ys |])

let train_x = batchSeqHistX true train
let test_x = batchSeqHistX false test

let train_y =                                 //for each engine-day y is the #of days remaining and 1 (for observed, i.e. not censored)
  train 
  |> Array.map(fun xs->xs.[0],xs.[1]) 
  |> Array.groupBy fst
  |> Array.map (fun (k,xs) -> k, xs |> Array.maxBy snd |> snd)
  |> Array.collect (fun (eng,mxd) -> [|for d in mxd .. -1.0f .. 1.0f-> [|d; 1.f|]|])  //engine data is not censored so 2nd is 1

let test_y =                                   //test y is the number of days to faliure 1 (observed)
  test_y_orig 
  |> Array.map(fun r->[|r.[0].AsFloat(); 1.0|]) 
      
(* validation
train_x.Length
train_y.Length
test_x.Length
test_y.Length

//validate data with python. use random index to compare e.g.:
train_x.[2000]
test_x.[97]
train_y.[230]
*)

let tte_mean_train = train_y |> Seq.averageBy (fun x->x.[0]) |> float
let mean_u = train_y |> Seq.averageBy (fun x->x.[1]) |> float

let init_alpha =
  let a = -1.0 /log(1.0 - 1.0/(tte_mean_train + 1.0))
  a/mean_u

let epsilon = 1e-10
let lowest_val = 1e-45
let scale_factor = 0.5
let max_beta = 10.0

let feature_dim = train_x.[0].[0].Length
let input_x = Node.Input(D feature_dim, dynamicAxes = [Axis.DefaultDynamicAxis(); Axis.DefaultBatchAxis()])
let input_y = Node.Input(D 2, dynamicAxes = [Axis.DefaultBatchAxis()])

//model uses this version
let weibull_loglik_discrete (ab_pred:Node) (y_true:Node) =
  let y_ = y_true.[0..1]
  let u_ = y_true.[1..2]
  let a_ = ab_pred.[0..1]
  let b_ = ab_pred.[1..2]

  let hazard0 = O.pow( O.abs((y_ + epsilon) ./ a_) , b_ )
  let hazard1 = O.pow( O.abs((y_ + 1.0) ./ a_ ), b_)

  let t = O.log(O.exp(hazard1 - hazard0) - (1.0 - epsilon))
  (u_ .* t) - hazard1
  

let weibull_loss clip_prob (ab_pred:Node) (y_true:Node) =
  let llh = weibull_loglik_discrete ab_pred y_true
  let loss = 
    match clip_prob with 
    | Some clip_prob ->
        O.clip(llh, Node.Scalar(log(clip_prob)), Node.Scalar(log(1.0 - clip_prob)) )
    | None -> llh
  -loss


let test_loss() =
  let a = 193.00
  let b = 3.5
  let y = 200.0
  let u = 1.0

  let abVar = Node.Input (D 2, dynamicAxes=[Axis.DefaultBatchAxis()])
  let yuVar = Node.Input(D 2, dynamicAxes=[Axis.DefaultBatchAxis()])
  let ll = weibull_loss None abVar yuVar
  let r = E.eval1 (dict[abVar.Var, [[a;b]] |> toValue; yuVar.Var, [[y;u]] |> toValue]) ll

  printfn "%A" r
  
let activateDebug (ab:Node) = ab

let activate2 (init_alpha:float) max_beta (scale_factor:float option) (ab:Node) =
  let a = ab.[0..1]
  let b = ab.[1..2]

  let scale_factor = defaultArg scale_factor 1.0

  let a = a .* scale_factor
  let b = b .* scale_factor
  let a = O.exp(a) .* init_alpha

  let b = 
    if max_beta > 1.05 then 
      let _shift = log(max_beta - (1.0 - epsilon))
      b - _shift
    else
      b
   
  let b = O.sigmod(b) .* max_beta
  
  //C.Splice(varVector [a.Var; b.Var], new Axis(-1)) |> F
  O.splice [a;b]

let activate1 (ab:Node) = 
  let a = ab.[0..1]
  let b = ab.[1..2]
  
  let a = O.exp(a)
  let b = O.softplus(b)
  
  O.splice [a;b]

let activate2_test () =
  let a = -0.0687800273 
  let b = -0.0426109992 
  let v = [[a;b]] |> toValue
  let ab = Node.Input(D 2, dynamicAxes=[Axis.DefaultBatchAxis()])
  let m = activate2 init_alpha max_beta (Some scale_factor) ab
  let o = m |> E.eval1 (dict[ab.Var,v])
  let a' = o.[0].[0]
  let b' = o.[0].[1]
  
  printfn "a':%f, b':%f" a' b'

let model_func = 
  let initState = new Constant(!> [| NDShape.InferredDimension |], dataType, 0.0) :> Variable |> V 
  let cell =  L.GRU2(D 20,activation=Activation.Tanh, enable_self_stabilization=false)
  L.Recurrence([initState]) cell       //recurrence layer needs a step function cell, GRU here
  >> O.getOutput 0
  >> O.last
  //>> L.Dropout(0.3)
  >> L.Dense(D 2)
  //>> activate1
  //>> activate4
  //>> activate3 max_beta
  //>> L.Activation Activation.
  >> activate2 init_alpha max_beta (Some scale_factor)
  //>> activate2 1.0 2.0 None

let mb_size = 200
let epoch_size = train_x.Length
let clip_loss = (Some 1e-6)

module Learners =
  let options = new AdditionalLearningOptions()
  options.gradientClippingThresholdPerSample <- 1e10
  options.gradientClippingWithTruncation <- true
  //options.l1RegularizationWeight <- 1e-10

  let adam model = 
    //let lr = constSchedule (0.01 / (float mb_size)) 
    let lr = new TrainingParameterScheduleDouble(0.001,1u)
    //let lrs = schedule [40, 0.001 / (float mb_size); 0, 0.0001 / (float mb_size) ]  mb_size
    let iter = epoch_size/mb_size
    let lrs = schedule [100*iter, 0.01; 50*iter, 0.01; 1, 0.00001]  mb_size
    let momentum = 0.9 //equivalent to beta1 in adam paper
    C.AdamLearner(
                      O.parms model |> parmVector
                      ,lr
                      ,new TrainingParameterScheduleDouble(momentum)                      
                      ,true                                     //should be exposed by CNTK C# API as C.DefaultUnitGainValue()
                      ,constSchedule 0.9999986111120757         //from python code, beta2 from adam paper
                      ,epsilon                                  //python defaults to 1e-8
                      ,false                                     //from python code
                      ,options
                      )

  let fsadagrad model = 
    //let lr = constSchedule (0.01 / (float mb_size)) 
    let lr = new TrainingParameterScheduleDouble(0.0001,1u)
    let iter = epoch_size/mb_size
    let lrs = schedule [2*iter, 0.0001; 1, 0.00001]  mb_size


    C.FSAdaGradLearner(
                      O.parms model |> parmVector
                      ,lr
                      ,constSchedule 0.9
                      ,true
                      ,constSchedule 0.9999986111120757
                      //,options
                      )


  let adadelta model = 
    //let lr = constSchedule (0.01 / (float mb_size)) 
    let lr = new TrainingParameterScheduleDouble(0.0000001,1u)
    let iter = epoch_size/mb_size
    let lrs = schedule [1*iter, 0.001; 1, 0.0001]  mb_size


    C.AdaDeltaLearner(
                      O.parms model |> parmVector
                      ,lr
                      ,0.99
                      ,1e-6
                      ,options
                      )

let createTrainer (model:Node) =

  let loglik_loss = weibull_loss clip_loss model input_y
  //let loglik_loss = -(weibull_loglik_discrete2 None model input_y)
  //let eval_f = weibull_loglik_discrete model input_y |> O.reduce_max [0]

  loglik_loss.Func.Save(@"D:\repodata\fscntk\wtte\loglik_loss.bin")
  //eval_f.Func.Save(@"D:\repodata\fscntk\wtte\eval_f.bin")
  model.Func.Save(@"D:\repodata\fscntk\wtte\wtte.bin")


  let parms = O.parms model 
  let totalParms = parms |> Seq.map (fun p -> p.Shape.TotalSize) |> Seq.sum
  printfn "Training %d parameters in %d parameter tensors" totalParms (Seq.length parms)
  printfn "detail"
  parms |> Seq.iter(fun p-> printfn "%s, %A, %d" p.Name (p.Shape.Dimensions) p.Shape.TotalSize)

  //let sq = O.squared_error (model,input_y)
  
  let learner = Learners.adam model

  let trainer = C.CreateTrainer(
                      model.Func,
                      loglik_loss.Func, 
                      //sq.Func,
                      null,
                      lrnVector [learner])
  trainer,learner


let trainModelTo (model:Node) (trainer:Trainer,learner:Learner) endEpoch =
  let x_batches = train_x |> Array.chunkBySize mb_size
  let y_batches = train_y |> Array.chunkBySize mb_size

  let rec loop epoch mb =
    let mb = mb % x_batches.Length
    let epoch = if mb = 0 then epoch + 1 else epoch
    let x = x_batches.[mb]
    let y = y_batches.[mb]

    let xSeq = x |> Array.map((Array.collect yourself)>>Array.toSeq)
    let xVal = Value.CreateBatchOfSequences(!-- (D feature_dim), xSeq, device)
    
    let ySeq = y |> Array.collect yourself |> Array.toSeq
    let yVal = Value.CreateBatch(!--(D 2), ySeq, device)

    //  //debugging
    //let evalInput = dict [input_x.Var,xVal; input_y.Var, yVal]
    //let m,outVar = trainer.LossFunction(),trainer.LossFunction().Output
    //let output = idict[outVar, (null:Value)]
    //m.Evaluate(evalInput,output,device)
    //let ePreds = output.[outVar]
    //let d = ePreds.GetDenseData<float32>(outVar) |> Seq.map Seq.toArray |> Seq.toArray
    //printfn "%A" d

    let input  = dict [input_x.Var, xVal; input_y.Var, yVal]
    let r = trainer.TrainMinibatch(input,false,device)
    let loss = trainer.PreviousMinibatchLossAverage()
    let eval = 0.0//trainer.PreviousMinibatchEvaluationAverage()
    let rate = learner.LearningRate()
    printfn "epoch: %d; mb=%d; loss=%f; eva=%f; rate=%f" epoch mb loss eval rate

    //let w = E.weights model
    //w |> Array.iter (printfn "%A")

    if epoch < endEpoch && not(Double.IsNaN(loss))then
      loop epoch (mb + 1)
  loop 0 0

let testModel (model:Node) =
  let xSeq = test_x |> Array.map((Array.collect yourself)>>Array.toSeq)
  let xVal = Value.CreateBatchOfSequences(!-- (D feature_dim), xSeq, device)
  let ySeq = test_y |> Array.collect yourself |> Array.toSeq |> Seq.map dmap
  let yVal = Value.CreateBatch(!--(D 2), ySeq, device)
  let m,outVar = model,model.Func.Output
  let evalInput = dict [input_x.Var,xVal]
  let output = idict[outVar, (null:Value)]
  m.Func.Evaluate(evalInput,output,device)
  let ePreds = output.[outVar]
  let d = ePreds.GetDenseData<float32>(outVar) |> Seq.map Seq.toArray |> Seq.toArray
  let loss = weibull_loss clip_loss m input_y
  let evalLoss = E.eval1 (dict[input_x.Var, xVal; input_y.Var, yVal]) loss
  let d2 = Array.zip3 d (evalLoss.[0]) test_y

  printfn "%A" d2
  printfn "test loss %f" (Seq.average evalLoss.[0])

let checkLoss() =
  let r = ([|107.161415f; 0.987768292f|], 5.73641109f, [|112.0; 1.0|])
  let ab,loss,yu = r
  let a = ab.[0] |> float
  let b = ab.[1] |> float
  let y = yu.[0] 
  let u = yu.[1] 

  let hazard0 = Math.Pow( (y + epsilon) / a , b )
  let hazard1 = Math.Pow ( (y + 1.0) / a , b)
 
  let llh = u * log(exp(hazard1 - hazard0) - (1.0 - epsilon)) - hazard1
  llh

let model = model_func input_x
let t,l = createTrainer model
trainModelTo model (t,l) 2
(*
trainModelTo (t,l) 3
trainModelTo model (t,l) 100 

let w = E.weights model
w |> Array.iter (printfn "%A")

[|[|107.161415f; 0.987768292f|]; [|103.427597f; 0.921889961f|];
  [|108.222733f; 0.919031441f|]; [|100.463837f; 0.931045711f|];
  test loss 5.397783, 5.414418
testModel model
-

*)

