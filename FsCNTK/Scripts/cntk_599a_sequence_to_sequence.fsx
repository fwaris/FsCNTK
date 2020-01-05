#load "FsCNTK_SetEnv.fsx"
open FsCNTK
open CNTK
open System.IO
open Blocks

//Reference tutorial: https://cntk.ai/pythondocs/CNTK_599A_Sequence_To_Sequence.html
//(the tutorial walks through different aspects of this problem and is therefore much longer;
//eventually it leads to the model presented here)

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

let vocab',i2w,w2i = get_vocab vocab_file

let input_vocab_size = vocab'.Length
let label_vocab_size = vocab'.Length

let sInput,sLabels = "S0","S1"
let streamConfigurations = 
    ResizeArray<StreamConfiguration>(
        [
            new StreamConfiguration(sInput, input_vocab_size, true)    
            new StreamConfiguration(sLabels, label_vocab_size,true)
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

//model dimension
let input_vocab_dim = input_vocab_size
let label_vocab_dim = label_vocab_size
let hidden_dim = 128
let rev_input = true
let isFast = false
let num_layers = 1

//*** need separate dynamic axes as the two sequences 
//are of different lengths
let inputAxis = Axis.NewUniqueDynamicAxis("inputAxis")
let labelAxis = Axis.NewUniqueDynamicAxis("labelAxis")

let raw_input = Node.Input
                    (
                        D input_vocab_dim, 
                        dynamicAxes = [inputAxis; Axis.DefaultBatchAxis()], //Sequence due to dynamic axis
                        name = "raw_input"
                    )

//label sequence
let raw_labels = Node.Input
                    (
                        D label_vocab_dim, 
                        dynamicAxes = [labelAxis; Axis.DefaultBatchAxis()],
                        name = "raw_labels"
                    )


//lstm layer helper
type LSTM_layer =
    static member create (input:Node, output_dim, ?recurrence_hook_h, ?recurrence_hook_c) = //optional args are available in type members
        let recurrence_hook_h = defaultArg recurrence_hook_h O.seq_past_value
        let recurrence_hook_c = defaultArg recurrence_hook_c O.seq_past_value

        let dh = Node.Placeholder(D output_dim, dynamicAxes = input.Var.DynamicAxes)
        let dc = Node.Placeholder(D output_dim, dynamicAxes = input.Var.DynamicAxes)

        let lstm_cell  = L.LSTM(D output_dim)

        let state = O.combine [dh; dc]
        let f_x_h_c = lstm_cell state input

        let h = f_x_h_c |> O.getOutput 0 |> recurrence_hook_h
        let c = f_x_h_c |> O.getOutput 1 |> recurrence_hook_c

        f_x_h_c.Func.ReplacePlaceholders(idict [dh.Var, h.Var; dc.Var, c.Var]) |> ignore

        f_x_h_c //O.combine [h; c]


let create_model() =
    let input_sequence = raw_input

    //head (singular value)
    let label_sentence_start = O.seq_first raw_labels

    //tail - for decoder training
    let label_sequence = O.seq_slice(raw_labels, 1, 0, name="label_sequence")

    //used as a mask
    let is_first_label = O.seq_is_first label_sequence

    //singular value (head) extended to a sequence of same length as tail
    let label_sentence_start_scattered = O.seq_scatter(label_sentence_start, is_first_label)

    //encoder
    let stabilize = B.Stabilizer()
    let output_h,output_c = 
        let prev_value_func = if rev_input then (*reverse sequence*) O.seq_future_value  else O.seq_past_value

        ((stabilize input_sequence, None),[for i in 1 .. num_layers -> i]) 
        ||> List.fold(fun (input,_) _ -> 

            let h_c = LSTM_layer.create(
                                    input, 
                                    hidden_dim, 
                                    recurrence_hook_h=prev_value_func, 
                                    recurrence_hook_c=prev_value_func)

            h_c |> O.getOutput 0, h_c |> O.getOutput 1 |> Some)

     //'thought' vectors from encoder
    let thought_vector_h = O.seq_first output_h
    let thought_vector_c = output_c |> Option.get |> O.seq_first

    let thought_vector_broadcast_h = O.seq_broadcast_as (thought_vector_h, label_sequence)
    let thought_vector_broadcast_c = O.seq_broadcast_as (thought_vector_c, label_sequence)

    //decoder
    let decoder_input = O.element_select(is_first_label, label_sentence_start_scattered, O.seq_past_value label_sequence)

    let decoder_output_h,decoder_output_c = 
        ((stabilize decoder_input, None),[for i in 0 .. num_layers-1 -> i]) 
        ||> List.fold(fun (input,_) i -> 
            let recurrence_hook_h, recurrence_hook_c =
                if i > 0 then
                    O.seq_past_value, O.seq_past_value
                else
                    (fun o -> O.element_select(is_first_label, thought_vector_broadcast_h, O.seq_past_value o)),
                    (fun o -> O.element_select(is_first_label, thought_vector_broadcast_c, O.seq_past_value o))

            let h_c = LSTM_layer.create(
                            input, 
                            hidden_dim, 
                            recurrence_hook_h=recurrence_hook_h, 
                            recurrence_hook_c=recurrence_hook_c)

            h_c |> O.getOutput 0, h_c |> O.getOutput 1 |> Some)
  
    let W = Node.Parm(Ds [O.shape decoder_output_h |> dims |> List.item 0; label_vocab_dim], C.GlorotUniformInitializer())
    let B = Node.Parm(D label_vocab_dim, init=0.)
    let stab_out = stabilize decoder_output_h
    let z = B + (stab_out * W)
    z.Func.Save(@"C:\s\repodata\fscntk\cntk_599\fs.bin")
    z

let model = create_model()

let label_sequence = model.Func.FindByName("label_sequence") |> F

let ce = O.cross_entropy_with_softmax(model, label_sequence)
let errs = O.classification_error(model, label_sequence)

for a in model.Func.Arguments do printfn "%A" a

let lr_per_sample = T.schedule_per_sample (0.007)
let minibatch_size = 72 
let momentum_schedule = T.schedule(0.9366416204111472, minibatch_size)
let clipping_threshold_per_sample = 2.3
let gradient_clipping_with_truncation = true

let learner = 
    let opts = new AdditionalLearningOptions()
    opts.gradientClippingThresholdPerSample <- clipping_threshold_per_sample
    opts.gradientClippingWithTruncation <- gradient_clipping_with_truncation
    Learner.MomentumSGDLearner(
            model |> O.parms |> parmVector,
            lr_per_sample,
            momentum_schedule,
            true,
            opts
            )

let trainer = Trainer.CreateTrainer(model.Func, ce.Func, errs.Func, ResizeArray[learner])

let training_progress_output_freq = 100
let max_num_minibatch = if isFast then 100 else 1000

let strInput = train_reader.StreamInfo(sInput)
let strLabels  = train_reader.StreamInfo(sLabels)
;;
for i in 1..max_num_minibatch do
    let mb_train = train_reader.GetNextMinibatch(uint32 minibatch_size, device)
    let args = idict [raw_input.Var,mb_train.[strInput]; raw_labels.Var,mb_train.[strLabels]]
    let r = trainer.TrainMinibatch(args,device)
    if i % training_progress_output_freq = 0 then
        let l =  trainer.PreviousMinibatchLossAverage()
        let c = trainer.PreviousMinibatchEvaluationAverage()
        printfn "Minibatch: %d, Train Loss: %0.3f Eval Crit.: %2.3f" i l c