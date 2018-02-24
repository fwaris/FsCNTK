﻿#load "..\Scripts\SetEnv.fsx"
open FsCNTK
open FsCNTK.FsBase
open FsCNTK.Layers
open Layers_Dense
open Layers_Dropout
open CNTK
open System.IO
open FsCNTK.FsBase
open Layers_Recurrence
open System.Collections.Generic
open Probability
open System
open FsCNTK.Layers_Recurrence

type CNTKLib = C
Layers.trace := true

let inputDim = 2000
let cellDim = 25
let hiddenDim = 25
let embeddingDim = 50
let numOutputClasses = 5

// build the model
let featuresName = "features";
let features = 
  Node.Input
    (
      D inputDim, 
      dynamicAxes=[Axis.DefaultDynamicAxis(); Axis.DefaultBatchAxis()],
      name=featuresName,
      isSparse=true)
  //Variable.InputVariable(new int[] { inputDim }, DataType.Float, featuresName, null, true /*isSparse*/);
let labelsName = "labels"
let labels = Node.Input(D numOutputClasses, dynamicAxes=[Axis.DefaultBatchAxis()],name=labelsName,isSparse=true)
  //Variable.InputVariable(new int[] { numOutputClasses }, DataType.Float, labelsName,
  //  new List<Axis>() { Axis.DefaultBatchAxis() }, true);

let model = 
  L.Embedding(D embeddingDim)
  >> L.Recurrence(L.LSTM(D hiddenDim, cell_shape=D cellDim,enable_self_stabilization=true),name="recurrence")
  >> List.head
  >> O.last
  >> L.Dense(D numOutputClasses,name="classifierOutput")

let pred = model features
let modelFile = @"D:\repodata\fscntk\Examplelstm_model_fs.bin"
pred.Func.Save(modelFile)
let _ = Function.Load(modelFile,device)

let loss = O.cross_entropy_with_softmax (pred,labels)
let lossFile = @"D:\repodata\fscntk\Examplelstm_loss_fs.bin"
loss.Func.Save lossFile
let _ = Function.Load(lossFile,device)

let ce = O.classification_error(pred,labels)

//input data streams 
let streamConfigurations = 
        [
            new StreamConfiguration(featuresName, inputDim, true, "x")    
            new StreamConfiguration(labelsName, numOutputClasses,false, "y")
        ]
        |> ResizeArray

let dataFolder = @"D:\Repos\cntk\Tests\EndToEndTests\Text\SequenceClassification\Data"
let trainFile = Path.Combine(dataFolder,"Train.ctf")

let minibatchSource = MinibatchSource.TextFormatMinibatchSource(
                        trainFile,
                        streamConfigurations,
                        MinibatchSource.InfinitelyRepeat,
                        true)

let featureStreamInfo = minibatchSource.StreamInfo(featuresName)
let labelsStreamInfo = minibatchSource.StreamInfo(labelsName)




//var classifierOutput = LSTMSequenceClassifierNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, "classifierOutput");
//var modelPath = Path.Combine(@"D:\repodata\fscntk", "ExampleLstm_model.bin");
//classifierOutput.Save(modelPath);
//var m = Function.Load(modelPath, device);
//var inputs = classifierOutput.Inputs;
//var parms = classifierOutput.Parameters();
//var outps = classifierOutput.Outputs;


