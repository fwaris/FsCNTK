#load "SetEnv.fsx"
open FsCNTK
open FsCNTK.FsBase
open FsCNTK.Layers
open Layers_Dense
open Layers_Dropout
open Layers_BN
open Layers_ConvolutionTranspose2D
open Layers_Convolution2D
open Layers_Sequence
open CNTK
open System.IO
open FsCNTK.FsBase
open System
(*
    Time series predicion with LSTM
    Based on: https://cntk.ai/pythondocs/CNTK_106A_LSTM_Timeseries_with_Simulated_Data.html
*)

let isFast = true

let split data val_size test_size =
    let ln = data |> Array.length |> float
    let pos_test = ln * (1. - test_size) |> int
    let pos_val = (float pos_test) * (1. - val_size) |> int
    let train,dval,test = data.[0..pos_val], data.[pos_val..pos_test], data.[pos_test..]
    Map.ofList ["train",train; "test",test; "val",dval] 
    
let generateData (fct:float32->float32) xs time_steps time_shift =
    let data =  xs |> Seq.map fct |> Seq.toArray
    let X = data |> Array.windowed time_steps 
    let Y = data |> Array.windowed (time_steps + time_shift) |> Array.map Array.last
    let minL = min X.Length Y.Length
    Array.zip (X |> Array.truncate minL) (Y |> Array.truncate minL)

let N = 5
let M = 5
let step= 0.010001000100010001f                 //Python code to get step --- (data,step) = np.linspace(0,100,10000,retstep=True,dtype=np.float32) 
let ts = [for i in 0.0f .. step .. 100.0f -> i]  
let data = generateData sin ts N M
let datasets = split data 0.1 0.1
datasets |> Map.iter (fun k v -> printfn "%s size: %d" k v.Length)

let create_model() =
    let init_state = Node.Const 0.1
    L.Recurrence [init_state;init_state] (L.LSTM(D N)) //number of states needs to be specified explicitly
    >> O.getOutput 0  //need to get output - full state is always returned
    >> O.seq_last
    >> L.Dropout(0.2, seed=1)
    >> L.Dense(D 1)

let TRAINING_STEPS = 10000
let BATCH_SIZE = 100
let EPOCHS = if isFast then 10 else 100

let batches (data:(float32[]*float32)[])  =
    data 
    |> Seq.chunkBySize BATCH_SIZE
    |> Seq.map (fun chnk -> 
        let x = Array.map fst chnk
        let y = Array.map snd chnk
        let xval = Value.CreateBatchOfSequences<float32>(!-- (D 1), Seq.cast<_> x, device)
        let yval = Value.CreateBatch(!-- (D 1), y, device)
        xval,yval)
    
datasets.["train"]  |>  batches |> Seq.item 0

let x = Node.Input (D 1, dynamicAxes=[ Axis.DefaultDynamicAxis(); Axis.DefaultBatchAxis()], name="x")
let z = create_model() x
let l = Node.Input (D 1, dynamicAxes=Seq.toList z.Var.DynamicAxes, name="y")

let learning_rate = 0.02
let lr_schedule = constSchedule learning_rate

let loss = O.squared_error(z,l)
let error = O.squared_error(z,l)

loss.Func.Save(@"C:\s\repodata\fscntk\lstm_ts\fs.bin")

let momentum_schedule = schedule [0,0.9] BATCH_SIZE

let learner = C.FSAdaGradLearner(z.Func.Parameters() |> parmVector, lr_schedule, momentum_schedule, true)

let trainer = C.CreateTrainer(z.Func, loss.Func, error.Func, lrnVector [learner])

for epoch in 1..EPOCHS do
    for (x1,y1) in batches datasets.["train"] do
        trainer.TrainMinibatch( idict [x.Var, x1; l.Var, y1], false, device) |> ignore
    if epoch % (EPOCHS / 10 ) = 0 then
        let training_loss = trainer.PreviousMinibatchLossAverage()
        printfn "epoch %d, loss: %0.5f" epoch training_loss
