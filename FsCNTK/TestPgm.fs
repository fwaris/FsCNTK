module Pgm
//for debugging only, this file not set to 'compile' normally
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
//See this tutorial for background documentation: 
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

let x = Node.Variable(D vocab_size, dynamicAxes = [Axis.DefaultDynamicAxis(); Axis.DefaultBatchAxis(); ])
let y = Node.Variable(D num_labels, dynamicAxes = [Axis.DefaultDynamicAxis(); Axis.DefaultBatchAxis(); ])

let create_model() =
  L.Embedding(shape=D emb_dim,name="embed")
  >> L.Recurrence(L.LSTM(D hidden_dim),go_backwards=false) // LSTM have two state variables
  >> List.head
  >> L.Dense(D num_labels,name="classify")


let z = create_model() x
//print(z.embed.E.shape)
//print(z.classify.b.value)

