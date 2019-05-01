#load "SetEnv.fsx"
open FsCNTK
open CNTK
open System.IO
open System
open Blocks

//************* model trains without error but needs more validation 

//Reference: https://cntk.ai/pythondocs/CNTK_204_Sequence_To_Sequence.html

type C = CNTKLib
Layers.trace := true

//Folder containing ATIS files which are
//part of the CNTK binary release download or CNTK git repo
let folder = @"c:\s\Repos\cntk\Examples\SequenceToSequence\CMUDict\Data"
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
let attention_span = 2
let attention_axis = new Axis(-3)
let use_attention = true
let use_embedding = true
let embedding_dim = 200
let length_increase = 1.5

let sentence_start = 
  let va = vocab' |> Array.map (function "<s>" -> 1.f | _ -> 0.f) 
  let vb = new NDArrayView(!-- (D vocab'.Length),va,device,true)
  new Constant(vb) :> Variable |> V

let sentence_end = 
  let va = vocab' |> Array.map (function "</s>" -> 1.f | _ -> 0.f) 
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
  let rec_layer f_step = L.Recurrence(f_step, initial_states=[cVal;cVal],go_backwards=gobk)
  let lstmCell() :StepFunction = L.LSTM(D hidden_dim, enable_self_stabilization=true)

  let inline ( !! ) f = f() //invoke a parameterless function (used to avoid too many braces)

  let LastRecurrence gb = 
    if not use_attention then 
      (fun f_step -> L.Fold (f_step, initial_states=[cVal;cVal], go_backwards=gb))
    else 
     rec_layer

  //encoder
  let encode =
    seq {
      yield embed
      yield B.Stabilizer(enable_self_stabilization=true)
      yield! seq {for i in 0 .. num_layers-1 -> rec_layer !!lstmCell}
      yield LastRecurrence gobk !!lstmCell
      yield O.mapOutputsZip [L.Label("encoded_h"); L.Label("encoded_c")] //lstm h and c outputs
    }
    |> Seq.reduce ( >> )

  let stab_in = B.Stabilizer(enable_self_stabilization=true)
  let rec_blocks = [for _ in 1 .. num_layers -> lstmCell() ]
  let stab_out = B.Stabilizer(enable_self_stabilization=true)
  let proj_out = L.Dense(D label_vocab_dim, name="out_proj")

  let attention_model = M.AttentionModel(D attention_dim, attention_span, attention_axis=attention_axis, name="attention_model")

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
      let head,tail = match rec_blocks with head::tail->head,tail | _ -> failwith "list empty"

      let r0  = head |>  if use_attention then lstm_with_attention >> rec_layer else rec_layer

      let rn  = tail |> List.map (fun rec_block -> 
        L.RecurrenceFrom(rec_block, num_states=2,go_backwards=gobk) encoded_input)

      (r0::rn) |> List.reduce ( >> )

    history
    |> (   embed 
        >> stab_in 
        >> rec_layers 
        >> O.getOutput 0
        >> stab_out 
        >> proj_out 
        >> L.Label("out_proj_out"))

  decode

let create_model_train (s2smodel:Node->Node->Node) input labels =
  //The input to the decoder always starts with the special label sequence start token.
  //Then, use the previous value of the label sequence (for training) or the output (for execution).
  let past_labels = L.Delay(initial_state=sentence_start) labels
  s2smodel past_labels input

let create_model_greedy s2smodel input =
  let unfold = 
    L.UnfoldFrom(  
      (fun history -> s2smodel history input |> O.hardmax),
      length_increase = length_increase,
      until_predicate = O.slice [0] [sentence_end_index] [sentence_end_index+1])
    
  unfold sentence_start input

let create_critetion_function model input labels = 
  let postprocessed_labels = O.seq_slice(labels,1,0) // <s> A B C </s> --> A B C </s>
  //let postprocessed_labels = O.reconcile_dynamic_axis(postprocessed_labels,input)
  let z = model input postprocessed_labels
  let ce = O.cross_entropy_with_softmax(z, postprocessed_labels)
  let errs = O.classification_error(z, postprocessed_labels)
  z,ce,errs

let train 
  (train_reader:MinibatchSource) 
  (valid_reader:MinibatchSource) 
  vocab 
  i2w 
  s2smodel 
  max_epochs 
  epoch_size 
  =
    //*** need separate dynamic axes as the two sequences 
    //are of different lengths
  let inputAxis = Axis.NewUniqueDynamicAxis("inputAxis")
  let labelAxis = Axis.NewUniqueDynamicAxis("labelAxis")

  let x = Node.Input
            (
              D input_vocab_dim, 
              dynamicAxes = [inputAxis; Axis.DefaultBatchAxis()] //Sequence due to dynamic axis
            )

  //label sequence
  let y = Node.Input
            (
              D label_vocab_dim, 
              dynamicAxes = [labelAxis; Axis.DefaultBatchAxis()] 
            )

  let model_train = create_model_train s2smodel 
  let z,ce,errs = create_critetion_function model_train x y

  let model_greedy = create_model_greedy s2smodel x

  model_greedy.Func.Save(@"C:\s\repodata\fscntk\cntk_204\fs.bin")

  let minibatch_size = 72
  let lr = if use_attention then 0.001 else 0.005

  let lr_per_sample = [lr; lr; lr/2.; lr/2.; lr/2.; lr/4.]

  let options = new AdditionalLearningOptions()
  options.gradientClippingThresholdPerSample <- 2.3
  options.gradientClippingWithTruncation <- true

  let learner = C.FSAdaGradLearner(
                      O.parms z |> parmVector ,
                      T.schedule_per_sample(lr_per_sample, epoch_size),
                      T.momentum_schedule(0.9366416204111472,minibatch_size),
                      true, //should be exposed by CNTK C# API as C.DefaultUnitGainValue()
                      T.momentum_schedule(0.9999986111120757), 
                      options)
  
  let trainer = C.CreateTrainer(
                      null,
                      ce.Func, 
                      errs.Func,
                      lrnVector [learner])

  let mutable total_samples = 0
  let mutable mbs = 0
  let eval_freq = 100
  let parms = O.parms z 
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
    printfn "Saving final model to '%s'" model_path
    model_greedy.Func.Save(model_path)
    printfn "%d epochs complete." max_epochs

let do_train() =
  train train_reader valid_reader vocab' () create_model 1 25000
;;
do_train()
