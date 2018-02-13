﻿#load "..\Scripts\SetEnv.fsx"
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

type CNTKLib = C

let inp = Node.Variable(D 58, dynamicAxes=[Axis.DefaultBatchAxis()]) 
let outp = Node.Variable(D 11, dynamicAxes=[Axis.DefaultBatchAxis()])

let model (inp:Node) = 
  let dim = 3
  let l1 = L.Dense(D dim,activation=Activation.Tanh) >> L.Dropout(dropout_rate=0.25) 
  let skipL =  L.Dense(D dim, activation=Activation.SELU) 
  ((inp |> l1 |> skipL) + (l1 inp)) |> L.Dense(D 11)


let pred = model inp

let loss = O.squared_error(pred,outp)

let lr = 0.0002
let momentum = 0.5 //equivalent to beta1

let learner = C.AdamLearner(
                    O.parms pred |> parmVector,
                    new TrainingParameterScheduleDouble(lr,1u),
                    new TrainingParameterScheduleDouble(momentum))

let trainer = C.CreateTrainer(pred.Func,loss.Func,null,lrnVector [learner])

let dataFile = "D:\gm\ca1.txt"

let inputSz = O.shape inp |> dims |> List.sum
let ouputSz = O.shape outp |> dims |> List.sum

let records = dataFile |> File.ReadAllLines |> Seq.length
let trainSz = float records * 0.70 |> int
let testSz = records - trainSz

let data = 
  let d =
    dataFile
    |> File.ReadAllLines
    |> Seq.map (fun s->s.Split([|'\t'|]) |> Array.filter (fun x->System.String.IsNullOrWhiteSpace x |> not))
    |> Seq.map (Seq.map float32>>Seq.toArray)
    |> Seq.toArray
  Array.shuffle d
  d

let testData = data |> Array.take trainSz
let trainData = data |> Array.skip trainSz

let X_train = trainData |> Array.map (Array.take inputSz) 
let Y_train = trainData |> Array.map (Array.skip inputSz) 
let X_test  = testData |> Array.map (Array.take inputSz)
let Y_test  = testData |> Array.map (Array.skip inputSz)

let X_trainBatch = Value.CreateBatch(!-- (O.shape inp), X_train |> Array.collect yourself, device)
let Y_trainBatch = Value.CreateBatch(!-- (O.shape outp), Y_train |> Array.collect yourself, device)
let X_testBatch  = Value.CreateBatch(!-- (O.shape inp), X_test |> Array.collect yourself, device)
let Y_testBatch  = Value.CreateBatch(!-- (O.shape outp), Y_test |> Array.collect yourself, device)

let train() =
  for i in 0..300000 do
      let inputs = idict [inp.Var, X_trainBatch; outp.Var, Y_trainBatch]
      let r = trainer.TrainMinibatch(inputs,false,device)
      if i%1000 = 0 then printfn "%d %A" i (trainer.PreviousMinibatchLossAverage())

let test() =
  let inputs = new UnorderedMapVariableMinibatchData()
  inputs.Add(inp.Var,new MinibatchData(X_testBatch))
  inputs.Add(outp.Var,new MinibatchData(Y_testBatch))
  let eval = C.CreateEvaluator(pred.Func)
  let r = eval.TestMinibatch(inputs)
  printfn "%A" r

let eval() =
  let eInp = idict [pred.Func.Arguments.[0],X_testBatch]
  let eOutp = idict [pred.Func.Outputs.[0],(null:Value)]
  pred.Func.Evaluate(eInp,eOutp,device)
  let ePreds = eOutp.[pred.Func.Outputs.[0]]
  let y' = ePreds.GetDenseData<float32>(outp.Var) |> Seq.map Seq.toArray |> Seq.toArray
  let y  = Y_test |> Seq.chunkBySize 11 |> Seq.toArray
  let yy = Array.zip y' (Array.collect yourself y)
  let fileData = yy |> Array.map (fun (a,b) -> let l=Array.zip a b |> Array.map (fun (a,b) -> sprintf "%f\t%f" a b) in System.String.Join("\t",l))
  do File.WriteAllLines(@"D:\gm\caOut.txt", fileData)
  let sz = y'.[0].Length
  let rms = Array.create sz 0.f 
  let sumSqrs =
    (rms,yy) 
    ||> Array.fold (fun acc (a,b) -> Array.zip a b |> Array.iteri(fun i (a,b) -> acc.[i] <-  acc.[i] + (a-b)*(a-b)); acc)
  let rmse = sumSqrs |> Array.map (fun x -> x / float32 Y_test.Length) |> Array.map sqrt 
  rmse

// Original
//43.35856
//32.6162
//24.24676
//11.40237
//19.8952
//22.47313
//28.90912
//23.52563
//21.65183
//17.47792
//10.53239

//RBF
//28
//20
//16.3
//11.9
//19.4
//23.2
//22
//19.06
//12.5
//12.4
//8.4

(*
let model = 
  L.Dense(D 20,activation=Activation.Tanh) 
  >> L.Dropout(dropout_rate=0.50) 
  >> L.Dense(D 20, Activation.Tanh)
  >> L.Dense(D 11)

 [|30.0817318f; 22.0587406f; 22.6269798f; 15.5886526f; 24.8051052f;
    28.8883686f; 30.0019913f; 19.628521f; 18.2731915f; 16.815958f; 9.77440929f|]

let model = 
  L.Dense(D 30,activation=Activation.Tanh) 
  >> L.Dropout(dropout_rate=0.50) 
  >> L.Dense(D 11)
val it : float32 [] =
  [|30.7224598f; 21.6095047f; 19.7907448f; 12.462801f; 16.1248436f;
    20.2330246f; 18.3850689f; 17.7195568f; 12.766511f; 16.5069218f;
    8.75440884f|]
let model = 
  L.Dense(D 30,activation=Activation.Tanh) 
  >> L.Dropout(dropout_rate=0.50) 
  >> L.Dense(D 10, activation=Activation.Tanh)
  >> L.Dense(D 10, activation=Activation.Tanh)
  >> L.Dense(D 11)
[|30.8770847f; 23.1529293f; 20.924551f; 18.078331f; 21.9063435f; 24.5609055f;
    25.8684177f; 21.0218277f; 12.6731205f; 13.5640898f; 10.1838083f|]
let model = 
  L.Dense(D 40,activation=Activation.Tanh) 
  >> L.Dropout(dropout_rate=0.50) 
  >> L.Dense(D 11)
  [|31.4304504f; 23.1725521f; 20.5174789f; 12.5884247f; 15.6073942f;
    18.2280674f; 17.6191349f; 17.5165787f; 12.0019941f; 16.1550407f;
    8.77223396f|]
let model = 
  L.Dense(D 50,activation=Activation.Tanh) 
  >> L.Dropout(dropout_rate=0.50) 
  >> L.Dense(D 11)
  [|31.9705982f; 22.9899044f; 19.8964729f; 12.7831888f; 15.9074898f;
    20.0760307f; 18.3155556f; 17.5104084f; 11.9928856f; 16.8134003f;
    9.06111717f|]
let model = 
  L.Dense(D 25,activation=Activation.Tanh) 
  >> L.Dropout(dropout_rate=0.50) 
  >> L.Dense(D 25,activation=Activation.Tanh) 
  >> L.Dense(D 11)
  [|36.6319122f; 26.5776272f; 21.7564716f; 15.6150675f; 21.5797176f;
    29.9412022f; 26.1687965f; 21.3152142f; 14.8019342f; 20.3547363f;
    10.8335228f|]
let model = 
  L.Dense(D 50,activation=Activation.Tanh) 
  >> L.Dropout(dropout_rate=0.50) 
  >> L.Dense(D 25,activation=Activation.Tanh) 
  >> L.Dense(D 11)
  [|36.5334854f; 25.5728321f; 21.0882835f; 16.7928867f; 23.7125378f;
    29.010458f; 27.3452015f; 24.0255318f; 15.3479137f; 20.9018707f;
    10.0443172f|]
let model = 
  L.Dense(D 11,activation=Activation.Tanh) 
  >> L.Dropout(dropout_rate=0.50) 
  >> L.Dense(D 11)
  [|30.4142151f; 20.9462662f; 18.9284115f; 13.8888073f; 18.7589569f;
    21.2629719f; 21.6855087f; 19.4213314f; 12.5832872f; 13.6198921f;
    8.8174715f|]
let model = 
  L.Dense(D 10,activation=Activation.Tanh) 
  >> L.Dropout(dropout_rate=0.50) 
  >> L.Dense(D 11)
  [|29.9408703f; 19.9023914f; 17.7270565f; 13.7511015f; 21.1314278f;
    24.3303471f; 25.1047897f; 21.347002f; 14.1011515f; 12.3981457f;
    9.04303551f|]
let model = 
  L.Dense(D 10,activation=Activation.Tanh) 
  >> L.Dropout(dropout_rate=0.25) 
  >> L.Dense(D 11)
  [|30.4716396f; 20.9844208f; 17.5037365f; 13.6963186f; 19.0177574f;
    21.3090286f; 22.1406593f; 20.8508568f; 12.9492798f; 12.5545006f;
    9.28462887f|]
let model = 
  L.Dense(D 5,activation=Activation.Tanh) 
  >> L.Dropout(dropout_rate=0.25) 
  >> L.Dense(D 11)
  [|27.3606853f; 19.8042774f; 17.7323227f; 13.7545738f; 18.6341114f;
    21.3458633f; 21.7192974f; 18.8949986f; 12.7396622f; 12.3491116f;
    9.31820202f|]  
let model = 
  L.Dense(D 7,activation=Activation.Tanh) 
  >> L.Dropout(dropout_rate=0.25) 
  >> L.Dense(D 11)
[|28.0927391f; 20.403038f; 18.071516f; 13.6054668f; 19.2439785f; 21.988903f;
    23.0910168f; 20.1121693f; 13.5672083f; 12.1169605f; 9.44756794f|]
let model (inp:Node) = 
  let dim = 3
  let l1 = L.Dense(D dim,activation=Activation.Tanh) >> L.Dropout(dropout_rate=0.25) 
  let skipL =  L.Dense(D dim, activation=Activation.SELU) 
  ((inp |> l1 |> skipL) + (l1 inp)) |> L.Dense(D 11)
 [|54.7022133f; 36.7688637f; 24.4712639f; 13.413847f; 37.1073418f;
    42.5674057f; 44.7219543f; 36.5981293f; 20.6601696f; 15.1724834f;
    11.1061411f|]

****
let model = 
  L.Dense(D 6,activation=Activation.Tanh) 
  >> L.Dropout(dropout_rate=0.25) 
  >> L.Dense(D 11)
  [|27.1436443f; 19.9475689f; 17.6807957f; 13.7557793f; 17.3315163f;
    19.5633659f; 19.8775024f; 18.1389008f; 12.0414429f; 12.3985281f;
    9.01100922f|]
   *)

   (*
train()
test()
eval()
*)

