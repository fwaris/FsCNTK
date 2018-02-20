
#load "..\Scripts\SetEnv.fsx"
open FsCNTK
open FsCNTK.FsBase
open FsCNTK.Layers
open Layers_Dense
open Layers_Dropout
open CNTK
open System.IO
open FsCNTK.FsBase
open System.Collections.Generic
open Probability
open System

type CNTKLib = C
let dataFile = @"D:\gm\ca1.txt"
let logFile =  @"F:\fwaris\CA1\runs.txt"

let inp = Node.Variable(D 58, dynamicAxes=[Axis.DefaultBatchAxis()]) 
let outp = Node.Variable(D 11, dynamicAxes=[Axis.DefaultBatchAxis()])

let inputSz = O.shape inp |> dims |> List.sum
let ouputSz = O.shape outp |> dims |> List.sum

let records = dataFile |> File.ReadAllLines |> Seq.length
let trainSz = float records * 0.50 |> int
let testSz = records - trainSz

let data = 
  let d =
    dataFile
    |> File.ReadAllLines
    |> Seq.map (fun s->s.Split([|'\t'|]) |> Array.filter (fun x->System.String.IsNullOrWhiteSpace x |> not))
    |> Seq.map (Seq.map float32>>Seq.toArray)
    |> Seq.toArray
  //Array.shuffle d
  d

let trainData = data |> Array.take trainSz
let testData = data |> Array.skip trainSz

Array.shuffle trainData

let X_train = trainData |> Array.map (Array.take inputSz) 
let Y_train = trainData |> Array.map (Array.skip inputSz) 
let X_test  = testData |> Array.map (Array.take inputSz)
let Y_test  = testData |> Array.map (Array.skip inputSz)

let X_trainBatch = Value.CreateBatch(!-- (O.shape inp), X_train |> Array.collect yourself, device)
let Y_trainBatch = Value.CreateBatch(!-- (O.shape outp), Y_train |> Array.collect yourself, device)
let X_testBatch  = Value.CreateBatch(!-- (O.shape inp), X_test |> Array.collect yourself, device)
let Y_testBatch  = Value.CreateBatch(!-- (O.shape outp), Y_test |> Array.collect yourself, device)

type MetaParms =
    {
        MaxLayers : int 
        Momentums : float list
        Activations : Activation list
        Dims        : int list
        Lr          : float list
    }

let hyperParms = 
    {
        MaxLayers = 4
        Momentums = [0.5; 0.75; 0.99]
        Activations = [Activation.Tanh; Activation.ELU; Activation.NONE; Activation.Sigmoid]
        Dims = [4; 5; 6; 7; 8; 9; 10]
        Lr = [0.0001; 0.0002; 0.0005]
    }

type ModelParms = 
    {
        LayerConfigs : (int*Activation*bool) list
        Lr : float
        Momentum : float
    }

let buildModel parms =
    let makeLayer (d,a,drop) =
        let l = L.Dense(D d, activation=a)
        if drop then l>>L.Dropout(dropout_rate=0.3) else l
    let model = 
        let layers = parms.LayerConfigs
        let (d1,a1,drop1) = List.head layers
        let l1 = layers |> List.head |> makeLayer
        (l1,List.tail layers) ||> List.fold (fun acc l -> acc >> makeLayer l)
    model >> L.Dense (D 11)

let genCandidateParms hyperP =
    let numLayers = RNG.Value.Next(1,hyperP.MaxLayers+1)
    let layers =  
        [
            for _  in 1 .. numLayers do
            let dims = hyperP.Dims.[ RNG.Value.Next(hyperP.Dims.Length)]
            let act = hyperP.Activations.[RNG.Value.Next(hyperP.Activations.Length)]
            let drop = RNG.Value.NextDouble() < 0.5
            yield (dims,act,drop)
        ]
    let momentum = hyperP.Momentums.[RNG.Value.Next(hyperP.Momentums.Length)]
    let lr = hyperP.Lr.[RNG.Value.Next(hyperP.Lr.Length)]
    {LayerConfigs=layers; Lr=lr; Momentum=momentum}

let rec genParms cache hyperP count =
    if count > 10000 then None
    else
        let p = genCandidateParms hyperP
        if Set.contains p cache then
            genParms cache hyperP (count + 1)
        else
            Some p

let writeLog parms trainLoss testRMSE runs =
    use fs = File.AppendText(logFile)
    fs.WriteLine(sprintf "%A\t%f\t%f\t%d" parms trainLoss testRMSE runs)

let tryModel parms =
    let pred = buildModel parms inp
    let loss = O.squared_error(pred,outp)
    let lr = parms.Lr
    let momentum = parms.Momentum

    let learner = C.AdamLearner(
                        O.parms pred |> parmVector,
                        new TrainingParameterScheduleDouble(lr,1u),
                        new TrainingParameterScheduleDouble(momentum))

    let trainer = C.CreateTrainer(pred.Func,loss.Func,null,lrnVector [learner])

    let eval() =
      let eInp = idict [pred.Func.Arguments.[0],X_testBatch]
      let eOutp = idict [pred.Func.Outputs.[0],(null:Value)]
      pred.Func.Evaluate(eInp,eOutp,device)
      let ePreds = eOutp.[pred.Func.Outputs.[0]]
      let y' = ePreds.GetDenseData<float32>(outp.Var) |> Seq.map Seq.toArray |> Seq.toArray
      let y  = Y_test |> Seq.chunkBySize 11 |> Seq.toArray
      let yy = Array.zip y' (Array.collect yourself y)
      //let fileData = yy |> Array.map (fun (a,b) -> let l=Array.zip a b |> Array.map (fun (a,b) -> sprintf "%f\t%f" a b) in System.String.Join("\t",l))
      //do File.WriteAllLines(@"D:\gm\caOut.txt", fileData)
      let sz = y'.[0].Length
      let rms = Array.create sz 0.f 
      let sumSqrs =
        (rms,yy) 
        ||> Array.fold (fun acc (a,b) -> Array.zip a b |> Array.iteri(fun i (a,b) -> acc.[i] <-  acc.[i] + (a-b)*(a-b)); acc)
      let rmse = sumSqrs |> Array.map (fun x -> x / float32 Y_test.Length) |> Array.map sqrt |> Seq.sum
      rmse

    let trainBatch() =
      let inputs = idict [inp.Var, X_trainBatch; outp.Var, Y_trainBatch]
      let r = trainer.TrainMinibatch(inputs,false,device)
      trainer.PreviousMinibatchLossAverage()

    let LOSS_COUNT = 1000
    let LOSS_TH = 0.05

    let lossesStagnating lossList =
        let h = List.head lossList
        let avgDiff = lossList |> List.tail |> List.map (fun a -> a - h |> abs) |> List.average
        avgDiff < LOSS_TH * h //less than 10%

    printfn "training model %A" parms
    let rec train prevLosses count =
        if count > 0 && count % 10000 = 0 then
            printfn "count=%d, loss=%f" count (trainer.PreviousMinibatchLossAverage())
        if count < 300000 then
            if List.length prevLosses < LOSS_COUNT then              
                train (trainBatch()::prevLosses) (count + 1)
            else
                if lossesStagnating prevLosses then 
                    printfn "loss stagnation %A after %d runs" prevLosses (trainer.TotalNumberOfSamplesSeen())
                    prevLosses.Head
                else
                    train (trainBatch()::prevLosses |> List.truncate LOSS_COUNT) (count + 1)
        else
            printfn "train count limit reached"
            prevLosses.Head
    
    let trainLoss = train [] 0
    let testRMSE = eval()
    let runs = trainer.TotalNumberOfSamplesSeen()

    writeLog parms trainLoss testRMSE runs
    printfn "%A, trainLoss=%f, RMSE=%f, Runs=%d" parms trainLoss testRMSE runs

let run() =
    let cache = Set.empty
    let go = ref true
    while !go do
        match genParms cache hyperParms 0 with
        | Some p -> tryModel p
        | None   -> go := false

(*
run()
*)