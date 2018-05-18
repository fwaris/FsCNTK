#load "SetEnv.fsx"
open FsCNTK
open FsCNTK.FsBase
open FsCNTK.Layers
open Layers_Dense
open Layers_BN
open Layers_ConvolutionTranspose2D
open Layers_Convolution2D
open Layers_Sequence
open Layers_Attention
open CNTK
open System.IO
open FsCNTK.FsBase
open System
open Blocks
open FsCNTK
open System.Drawing

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
  let rec_layer() = L.Recurrence(go_backwards=gobk)
  let lstmCell() :StepFunction = L.LSTM(D hidden_dim, enable_self_stabilization=true)

  let inline ( !! ) f = f() //invoke a parameterless function (used to avoid too many braces)

  let LastRecurrence gb = 
    if not use_attention then 
      L.Fold (go_backwards=gb)
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

  let attention_model = L.AttentionModel(D attention_dim, name="attension_model")

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
        L.RecurrenceFrom(go_backwards=gobk) rec_block encoded_input)

      (r0::rn) |> List.reduce ( >> )

    history
    |> (   embed 
        >> stab_in 
        >> rec_layers 
        >> stab_out 
        >> proj_out 
        >> L.Label("out_proj_out"))

  decode

let create_model_train (s2smodel:Node->Node->Node) input labels =
  //The input to the decoder always starts with the special label sequence start token.
  //Then, use the previous value of the label sequence (for training) or the output (for execution).
  let past_labels = L.Delay(sentence_start) labels
  s2smodel past_labels input

let create_model_greedy s2smodel input =
  let unfold = 
    L.UnfoldFrom(
      length_increase = length_increase,
      until_predicate = O.slice [0] [sentence_end_index] [sentence_end_index]) //python: lambda w:w[...,sentence_end_index]  - gets the indexed value at last dimension
      (fun history -> s2smodel history input |> O.hardmax)
  unfold sentence_start input

let create_critetion_function model input labels = 
  let postprocessed_labels = O.seq_slice(labels,1,0) // <s> A B C </s> --> A B C </s>
  let z = model input postprocessed_labels
  let ce = O.cross_entropy_with_softmax(z, postprocessed_labels)
  let errs = O.classification_error(z, postprocessed_labels)
  ce,errs

let train 
  (train_reader:MinibatchSource) 
  (valid_reader:MinibatchSource) 
  vocab 
  i2w 
  s2smodel 
  max_epochs 
  epoch_size 
  =
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

  let model_train = create_model_train s2smodel x y
  let ce,errs = create_critetion_function s2smodel x y

  let model_greedy = create_model_greedy s2smodel x

  let minibatch_size = 72
  let lr = if use_attention then 0.001 else 0.005

  //lr = C.learning_parameter_schedule_per_sample([lr]*2+[lr/2]*3+[lr/4], epoch_size),
  //momentum = C.momentum_schedule(0.9366416204111472, minibatch_size=minibatch_size),
  //gradient_clipping_threshold_per_sample=2.3,
  //gradient_clipping_with_truncation=True)

  let lr_per_sample = [[lr;lr]; [for _ in 1..3->lr/2.]; [for _ in 1..4->lr/4.]] |> List.collect yourself
  let lr_per_minibatch = lr_per_sample |> List.mapi (fun i r -> i, r * float minibatch_size)
  let lr_schedule = schedule lr_per_minibatch epoch_size

  let momentum = new TrainingParameterScheduleDouble(0.9366416204111472,uint32 minibatch_size)

  let options = new AdditionalLearningOptions()
  options.gradientClippingThresholdPerSample <- 2.3
  options.gradientClippingWithTruncation <- true

  let learner = C.FSAdaGradLearner(
                      O.parms model_train |> parmVector ,
                      lr_schedule,
                      momentum,
                      true,                                   //should be exposed by CNTK C# API as C.DefaultUnitGainValue()
                      null, //not sure what should be the default
                      options)
  
  let trainer = C.CreateTrainer(
                      model_train.Func,
                      ce.Func, 
                      errs.Func,
                      lrnVector [learner])

  let mutable total_samples = 0
  let mutable mbs = 0
  let eval_freq = 100
  let parms = O.parms model_train 
  let totalParms = parms |> Seq.map (fun p -> p.Shape.TotalSize) |> Seq.sum
  printfn "Training %d parameters in %d parameter tensors" totalParms (Seq.length parms)

  let strInput = train_reader.StreamInfo(sInput)
  let strLabels  = train_reader.StreamInfo(sLabels)

  for epoch in 1 .. max_epochs do
    while total_samples < (epoch+1) * epoch_size do
      let mb_train = train_reader.GetNextMinibatch(uint32 minibatch_size, device)
      let args = idict [x.Var,mb_train.[strInput]; y.Var,mb_train.[strLabels]]
      let r = trainer.TrainMinibatch(args,device)

      if mbs % eval_freq = 0 then
        let mb_valid = valid_reader.GetNextMinibatch(1u)
        let inpStr = mb_valid.[strInput]
        let e = E.eval inpStr model_greedy
        //vizualization code to be added later
        ()

      total_samples <- total_samples + (int mb_train.[strLabels].numberOfSamples)
      mbs <- mbs + 1
      
    let model_path = sprintf @"D:\repodata\fscntk\s2s\model_%d.cmf" epoch
    printf "Saving final model to '%s'" model_path
    model_train.Func.Save(model_path)
    printf "%d epochs complete." max_epochs
