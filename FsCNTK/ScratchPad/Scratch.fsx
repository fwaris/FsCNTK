﻿#load "..\Scripts\SetEnv.fsx"
open FsCNTK
open FsCNTK.FsBase
open Layers_Dense
open Layers_Sequence
open Models_Attention
open CNTK
open System.IO
open System
open Blocks
open System
open FsCNTK
open FsCNTK

type C = CNTKLib
Layers.trace := true

//Folder containing ATIS files which are
//part of the CNTK binary release download or CNTK git repo
let folder = @"D:\Repos\cntk\Examples\SequenceToSequence\CMUDict\Data"
let place s = Path.Combine(folder,s)

let validation = place "tiny.ctf"
let training = place "cmudict-0.7b.train-dev-20-21.ctf"
let testing = place "cmudict-0.7b.test.ctf"
let vocab_file = place "cmudict-0.7b.mapping"

let get_vocab path =
  let vocab = path |> File.ReadLines |> Seq.map (fun s->s.Trim()) |> Seq.toArray
  let i2w = vocab |> Seq.mapi (fun i w -> i,w) |> Map.ofSeq
  let w2i = vocab |> Seq.mapi (fun i w -> w,i) |> Map.ofSeq
  vocab,i2w,w2i

let input_vocab_dim = 69
let label_vocab_dim = 69

let vocab',i2w,w2i = get_vocab vocab_file

let sInput,sLabels = "S0","S1"
let streamConfigurations = 
    ResizeArray<StreamConfiguration>(
        [
            new StreamConfiguration(sInput, input_vocab_dim, true)    
            new StreamConfiguration(sLabels, label_vocab_dim,true)
        ]
        )

let  create_reader path is_training =
  MinibatchSource.TextFormatMinibatchSource
    (
      path,
      streamConfigurations,
      (if is_training then  MinibatchSource.InfinitelyRepeat else MinibatchSource.FullDataSweep),
      is_training
    )

//Train data reader
let train_reader = create_reader training true 

//Validation data reader
let valid_reader = create_reader validation true

let hidden_dim = 512
let num_layers = 2
let attention_dim = 128
let use_attention = true
let use_embedding = true
let embedding_dim = 200
let length_increase = 1.5

let sentence_start = 
  let va = vocab' |> Array.map (function "<s>" -> 1.f | _ -> 0.f) 
  let vb = new NDArrayView(!-- (D vocab'.Length),va,device,true)
  new Constant(vb) :> Variable |> V

let  sentence_end_index = Array.IndexOf(vocab',"</s>")

//The main structure we want: https://cntk.ai/jup/cntk204_s2s3.png
let create_model =

  let embed =
    if use_embedding then
      L.Embedding(D embedding_dim, name="embed")
    else
      O.identity
  
  let gobk = not use_attention

  //recurrent layer and cell creation helpers that encapsulate defaults
  let cVal = new Constant(!> [| NDShape.InferredDimension |], dataType, 0.1) :> Variable |> V 
  let rec_layer() = L.Recurrence(initial_states=[cVal;cVal],go_backwards=gobk)
  let lstmCell() :StepFunction = L.LSTM(D hidden_dim, enable_self_stabilization=true)

  let inline ( !! ) f = f() //invoke a parameterless function (used to avoid too many braces)

  let LastRecurrence gb = 
    if not use_attention then 
      L.Fold (initial_states=[cVal;cVal], go_backwards=gb)
    else 
     !!rec_layer

  //encoder
  let encode =
    seq {
      yield embed
      yield B.Stabilizer(enable_self_stabilization=true)
      yield! seq {for i in 0 .. num_layers-1 -> !!rec_layer !!lstmCell}
      yield LastRecurrence gobk !!lstmCell
      yield O.mapOutputsZip [L.Label("encoded_h"); L.Label("encoded_c")] //lstm h and c outputs
    }
    |> Seq.reduce ( >> )

  let stab_in = B.Stabilizer(enable_self_stabilization=true)
  let rec_blocks = [for _ in 1 .. num_layers -> lstmCell() ]
  let stab_out = B.Stabilizer(enable_self_stabilization=true)
  let proj_out = L.Dense(D label_vocab_dim, name="out_proj")

  let attention_model = M.AttentionModel(D attention_dim, name="attension_model")

  let decode history input =

    let encoded_input = encode(input)

    //compose lstm with attention
    let lstm_with_attention (fn:StepFunction) : StepFunction =
      fun (dhdc:Node) (x:Node) ->
        let dh = O.getOutput 0 dhdc 
        let h_att = attention_model (O.getOutput 0 encoded_input, dh)
        let x = O.splice [x; h_att]  
        fn dhdc x

    let rec_layers = 
      let head::tail = rec_blocks

      let r0  = head |>  if use_attention then lstm_with_attention >> !!rec_layer else !!rec_layer

      let rn  = tail |> List.map (fun rec_block -> 
        L.RecurrenceFrom(num_states=2,go_backwards=gobk) rec_block encoded_input)

      (r0::rn) |> List.reduce ( >> )

    history
    |> (   embed 
        >> stab_in 
        >> rec_layers 
        >> stab_out 
        >> proj_out 
        >> L.Label("out_proj_out"))

  decode


let cVal = new Constant(!> [| NDShape.InferredDimension |], dataType, 0.1) :> Variable |> V 
let e1 = L.Recurrence([cVal;cVal]) (L.LSTM(D hidden_dim))
let e2 = L.Recurrence([cVal;cVal]) (L.LSTM(D hidden_dim))
let encoder = e1 >> e2

let x = Node.Input
          (
            D input_vocab_dim, 
            dynamicAxes = [Axis.DefaultDynamicAxis(); Axis.DefaultBatchAxis()] //Sequence due to dynamic axis
          )

//label sequence
let y = Node.Input
          (
            D label_vocab_dim, 
            dynamicAxes = [Axis.DefaultDynamicAxis(); Axis.DefaultBatchAxis()] 
          )

//let strInput = train_reader.StreamInfo(sInput)
//let strLabels  = train_reader.StreamInfo(sLabels)
//let vr = valid_reader.GetNextMinibatch(1u).[strInput]
//let v = E.eval vr m
//let vv = v.[m.Func.Outputs.[0]]
//let v1h = vv.GetDenseData<float32>(m.Func.Outputs.[0]) 

let encoded_input = encoder x

let attention_model = M.AttentionModel(D attention_dim, name="attension_model")

let lstm_with_attention (fn:StepFunction) : StepFunction =
  fun (dhdc:Node) (x:Node) ->
    let dh = O.getOutput 0 dhdc 
    let h_att = attention_model (O.getOutput 0 encoded_input, dh)
    let x = O.splice [x; h_att]  
    fn dhdc x


let datt = lstm_with_attention (L.LSTM(D hidden_dim))
let d1 = L.Recurrence([cVal;cVal]) datt
let d2 = L.RecurrenceFrom(2) (L.LSTM(D hidden_dim)) encoded_input
let proj_out = L.Dense(D label_vocab_dim, name="out_proj")

let decoder = d1 >> d2 >> O.getOutput 0 >> proj_out

let m = decoder x
