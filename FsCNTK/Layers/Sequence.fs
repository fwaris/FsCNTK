namespace FsCNTK
open CNTK
open System
open FsBase
open Blocks
open Layers

#nowarn "25"

module Layers_Sequence =
  
  type private Cell = LSTM | GRU | RNNStep
          
  type StepFunction = 
    //Shape list (*shapes of states*) * 
    (Node (*states [combined] *) -> Node (* input *) -> Node (* combined outputs *)) // function signature

  type private RParms =
    {
        //shape related
        stack_axis            : AxisVector
        stacked_dim           : int
        cell_shape_stacked    : Shape
        cell_shape_stacked_H  : Shape
        cell_shape            : Shape
        //stabilizations
        Sdh                   : Node -> Node 
        Sdc                   : Node -> Node
        //Sct                   : Node -> Node
        Sht                   : Node -> Node
        //parameters
        b                     : Node
        W                     : Node
        H                     : Node
        H1                    : Node option
        //output projection
        Wmr                   : Node option
    }

  type L with

    static member private delay (x:Node,initial_state:Node option, time_step, name) =
      //let initial_state = 
      //  match initial_state with 
      //  | Some (v:Node) -> v.Var 
      //  | None -> Node.Variable(D NDShape.InferredDimension, kind=VariableKind.Constant).Var

      let out =
        if time_step > 0 then
          match initial_state with
          | Some v  -> C.PastValue(x.Var, v.Var,uint32 time_step,name) 
          | None    -> C.PastValue(x.Var, uint32 time_step,name)
        elif time_step < 0 then
          match initial_state with
          | Some v -> C.FutureValue(x.Var,v.Var,uint32 -time_step,name)
          | None -> C.FutureValue(x.Var,uint32 -time_step,name)
        else
          if name="" then
            x.Func
          else
            C.Alias(x.Var,name) 
      F out
        

    static member Delay(?initial_state:Node, ?time_step, ?name) =
      let time_step = defaultArg time_step 1
      let name = defaultArg name ""
      fun (x:Node) -> L.delay (x,initial_state,time_step,name)


    static member private RecurrentBlock
      (
        cellType,
        out_shape,
        cell_shape,
        enable_self_stabilization,
        (init:CNTKDictionary),
        (init_bias:float),
        (x:Node)
      )
      =      
      let has_projection,cell_shape = 
        match cell_shape with 
        | Shape.Unknown -> false,out_shape 
        | c             -> true,c //what if the cell shape is same as output?
      
      let Wmr = 
          if has_projection 
          then  Node.Parm(cell_shape + out_shape, init=init, name="P") |> Some
          else None

      if len out_shape <> 1 || len cell_shape <> 1 then
        failwithf "%A shape and cell_shape must be vectors" cellType

      //see python code and comment in blocks.py
      //for stacking of multiple variables along the fastest changing
      //dimension for efficient computation 
      let cell_dim = cell_shape |> dims |> List.last // or first as only 1 dimension

      let stacked_dim = cell_dim * match cellType with 
                                      | RNNStep ->  1 
                                      | GRU     ->  3 //3 gates
                                      | LSTM    ->  4 //4 gates

      let cell_shape_stacked = D stacked_dim

      let stacked_dim_h = cell_dim * match cellType with 
                                        | RNNStep ->  1 
                                        | GRU     ->  2 
                                        | LSTM    ->  4

      let cell_shape_stacked_H  = D stacked_dim_h

      let Sdh = B.Stabilizer(enable_self_stabilization=enable_self_stabilization, name="dh_stabilizer")
      let Sdc = B.Stabilizer(enable_self_stabilization=enable_self_stabilization, name="dc_stabilizer")
      //let Sct = L.Stabilizer(enable_self_stabilization=enable_self_stabilization, name="c_stabilizer") //for peepholes
      let Sht = B.Stabilizer(enable_self_stabilization=enable_self_stabilization, name="P_stabilizer")

      let b = Node.Parm(               cell_shape_stacked,   init=init_bias, name="b")
      let W = Node.Parm(   O.shape x + cell_shape_stacked,   init=init, name="W")
      let H = Node.Parm(   out_shape + cell_shape_stacked_H, init=init, name="H")

      let H1 = 
        match cellType with 
        | GRU -> Node.Parm(out_shape + cell_shape          , init=init, name="H1") |> Some
        | _   -> None

      {
          stack_axis            = axisVector [new Axis(0)] 
          stacked_dim           = cell_dim
          cell_shape_stacked    = cell_shape_stacked
          cell_shape_stacked_H  = cell_shape_stacked_H
          cell_shape            = cell_shape
          Sdh                   = Sdh 
          Sdc                   = Sdc
          //Sct                   = Sct
          Sht                   = Sht
          b                     = b
          W                     = W
          H                     = H
          H1                    = H1
          Wmr                   = Wmr
      }

    static member LSTM
      (
        out_shape,
        ?cell_shape,
        ?activation,
        ?enable_self_stabilization,
        ?init,
        ?init_bias,
        //?use_peephole, - not that useful according to "LSTM: A Search Space Odyssey", https://arxiv.org/pdf/1503.04069.pdf
        ?name
      )
      : StepFunction
      =
      let activation = defaultArg activation Activation.Tanh
      let init = defaultArg init (C.GlorotUniformInitializer())
      let init_bias = defaultArg init_bias 0.0
      let enable_self_stabilization = defaultArg enable_self_stabilization false
      let name = defaultArg name ""


      //Great reference: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
      let lstm states (x:Node)   =
        let dh,dc =
          match O.uncombine states with
          | []        -> failwithf "expect initial state"
          | a::b::_   -> a,b
          | [a]       -> a,a

        let rp =
            L.RecurrentBlock
              (
                  Cell.LSTM, 
                  out_shape,
                  defaultArg cell_shape Shape.Unknown,
                  enable_self_stabilization,
                  init,
                  init_bias,
                  x
              )

        let dhs = rp.Sdh(dh) //stabilized previous output 
        //let dcs = rp.Sdc(dc) //stabilized previous cell state only needed if peepholes are used

        //projected contribution from inputs(s), hidden and bias
        let wProj = (x * rp.W) 
        let hProj = (dhs * rp.H) 
        let proj4 = (rp.b + wProj) + hProj

          //rp.b 
          //+  (x * rp.W) 
          //+ (dhs * rp.H) 
          
        let it_proj  = proj4 |> O.slice rp.stack_axis [0*rp.stacked_dim] [1*rp.stacked_dim]  // split along stack_axis
        let bit_proj = proj4 |> O.slice rp.stack_axis [1*rp.stacked_dim] [2*rp.stacked_dim]
        let ft_proj  = proj4 |> O.slice rp.stack_axis [2*rp.stacked_dim] [3*rp.stacked_dim]
        let ot_proj  = proj4 |> O.slice rp.stack_axis [3*rp.stacked_dim] [4*rp.stacked_dim]

        let it  = O.sigmod it_proj                          //input gate(t)
        let bit = it .* (L.Activation activation bit_proj)  //applied to tanh of input network
        let ft  = O.sigmod ft_proj                          //forget-me-not gate(t)
        let bft = ft .* dc                                  //applied to cell(t-1)
        let ct  = bft + bit                                 //c(t) is sum of both
        let ot  = O.sigmod ot_proj                          //output gate(t)
        let ht  = ot .* (L.Activation activation ct)        //applied to tanh(cell(t))

        let c = ct                                          //cell value
        let h = match rp.Wmr with Some w -> (rp.Sht ht) * w | None -> ht //if has_projection then C.Times(Wmr, !>Sht(F ht)) else ht

        O.combine [h;c]

      let h_shape = out_shape
      let c_shape = defaultArg cell_shape out_shape

      //step_function
      lstm
      //[h_shape; c_shape], lstm 

    static member GRU
      (
        out_shape,
        ?cell_shape,
        ?activation,
        ?init,
        ?init_bias,
        ?enable_self_stabilization,
        ?name
      )
      : StepFunction
      =
      let activation = defaultArg activation Activation.Tanh
      let init = defaultArg init (C.GlorotUniformInitializer())
      let init_bias = defaultArg init_bias 0.0
      let enable_self_stabilization = defaultArg enable_self_stabilization false
      let name = defaultArg name ""

      let gru dh (x:Node)  =

        let rp =
            L.RecurrentBlock
              (
                  Cell.GRU, 
                  out_shape,
                  defaultArg cell_shape Shape.Unknown,
                  enable_self_stabilization,
                  init,
                  init_bias,
                  x
              )

        let dhs = rp.Sdh(dh) //previous value stabilized
        let projx3 = rp.b + (x * rp.W) 
        let projh2 = dhs * rp.H 

        let zt_proj = (projx3 |> O.slice rp.stack_axis [0*rp.stacked_dim] [1*rp.stacked_dim])
                      +
                      (projh2 |> O.slice rp.stack_axis [0*rp.stacked_dim] [1*rp.stacked_dim])

        let rt_proj = (projx3 |> O.slice rp.stack_axis [1*rp.stacked_dim] [2*rp.stacked_dim])
                      +
                      (projh2 |> O.slice rp.stack_axis [1*rp.stacked_dim] [2*rp.stacked_dim])

        let ct_proj =  projx3 |> O.slice rp.stack_axis [2*rp.stacked_dim] [3*rp.stacked_dim]


        let zt = O.sigmod zt_proj   // update gate z(t)
        let rt = O.sigmod rt_proj   // reset gate r(t)
        let rs = dhs .* rt          // "cell" c

        let ct = L.Activation activation (ct_proj + (rs * rp.H1.Value)) 

        //Python:  ht = (1 - zt) * ct + zt * dhs
        let ht = (1. - zt) .* ct + (zt .* dhs) 

        let h = match rp.Wmr with Some w -> (rp.Sht ht) * w | None -> ht 

        h
      
      //step_function
      //[out_shape],gru
      gru


    static member RnnStep
      (
        out_shape,
        ?cell_shape,
        ?activation,
        ?init,
        ?init_bias,
        ?enable_self_stabilization,
        ?name
      )
      : StepFunction
      =
      let activation = defaultArg activation Activation.Tanh
      let init = defaultArg init (C.GlorotUniformInitializer())
      let init_bias = defaultArg init_bias 0.0
      let enable_self_stabilization = defaultArg enable_self_stabilization false
      let name = defaultArg name ""

      let rnn_step dh (x:Node) =

        let rp =
            L.RecurrentBlock
              (
                  Cell.RNNStep, 
                  out_shape,
                  defaultArg cell_shape Shape.Unknown,
                  enable_self_stabilization,
                  init,
                  init_bias,
                  x
              ) 
        let dhs = rp.Sdh(dh)

        let ht = L.Activation activation (rp.b + (x * rp.W) + (dhs * rp.H))
        let h = match rp.Wmr with Some w -> (rp.Sht ht) * w | None -> ht //if has_projection then C.Times(Wmr, !>Sht(F ht)) else ht

        h

      //[out_shape],rnn_step
      rnn_step

    static member RecurrenceFrom 
      (
       num_states : int, //number of states used by the step function, >= 1
       ?go_backwards,
       ?name
      )
      =
      let go_backwards = defaultArg  go_backwards false //'false' means get past value - somewhat counter intuitive but matches python
      let name = defaultArg name ""

      fun (func:StepFunction) states (x:Node) ->

        let time_step = if go_backwards then -1 else +1

        let states = O.uncombine states

        let x = O.getOutput 0 x // peel off the first output (as RNN outputs have combined output/states)
        //let cVal = new Constant(!> [| NDShape.InferredDimension |](*!>[|1|]*), dataType, 0.0) :> Variable |> V 
        ////fix the states to match the number of expected states
        //let states1 = [cVal; cVal]
        //let states = [states.[0]; states.[0]]
        let states = 
          match List.length states, num_states with
          | 1,x when states.[0].Var.IsConstant -> [for i in 1..x -> states.[0]]
          | a,b when a = b -> states
          | _  -> failwith "number of input states should match what is required by step function or be a constant"

        //placeholder for each state variable
        let out_vars_fwd = states |> List.map (fun s -> Node.Placeholder(O.shape s,x.Var.DynamicAxes))

        //placeholders run through the delay function for prior (or future) values
        let prev_out_vars = 
          List.zip out_vars_fwd states 
          |> List.map (fun (ph,st) ->  L.delay(ph,Some st,time_step,""))
          |> O.combine

        //call the step function with delayed values
        let out = func prev_out_vars x

        //loop - replace placeholders with step function output to close the loop
        let out_vars = 
          List.zip out_vars_fwd (O.uncombine out)
          |> List.map (fun (fw,ac) -> 
            ac.Func.ReplacePlaceholders(idict[fw.Var,ac.Var]) |> F )

        //return step function output
        out

    static member Recurrence
      (
       initial_states,
       ?go_backwards,
       ?name
      )                            
      =
      if List.isEmpty initial_states then failwithf "Recurrence: init_states list cannot be empty. The number of initial states should match the number expected by the step function"

      fun (func:Node->Node->Node) (x:Node) -> 
        let go_backwards = defaultArg  go_backwards false
        let name = defaultArg name ""
        //let state_shapes,_ = step_function
      
        let recurrence_from = L.RecurrenceFrom(List.length initial_states, go_backwards=go_backwards,name=name) func

        if !Layers.trace then printfn ">> Recurrence with %A" (initial_states |> List.map (O.shape>>dims))
        recurrence_from (O.combine initial_states) x

    static member Fold
      (
       initial_states,
       ?go_backwards,
       ?name
      )                            
      =

      fun  (func:Node->Node->Node) (x:Node) -> 
        let go_backwards = defaultArg  go_backwards false
        let name = defaultArg name ""
        //let state_shapes,_ = step_function

        let recurrence = L.Recurrence(
                              initial_states,
                              go_backwards=go_backwards, 
                              name=name) 
                            func

        let get_final = if go_backwards then O.seq_first else O.seq_last
        
        (recurrence >> (O.mapOutputs get_final)) x

    static member UnfoldFrom
      (
       ?until_predicate:(Node->Node),
       ?length_increase,     
       ?name
      )                            
      =
      let length_increase = defaultArg length_increase 1.0
      let name = defaultArg name ""

      fun generator_function initial_state dynamic_axes_like ->
        
        let out_axis = 
          if length_increase = 1.0 then
            dynamic_axes_like
          else
            let factors = O.seq_broadcast_as(Node.Scalar length_increase,dynamic_axes_like)
            O.seq_where factors 

        let states = O.uncombine initial_state
        let out_vars_fwd = states |> List.map (fun s -> Node.Placeholder(O.shape s))

        //placeholders run through the delay function for prior values
        let prev_out_vars = 
          List.zip out_vars_fwd states 
          |> List.map (fun (ph,st) ->  L.delay(ph,Some st,1,""))
          |> O.combine

        //generator function emits output and optionally new state
        let z = generator_function prev_out_vars 

        let output = O.getOutput 0 z
      
        let newState = 
          if z.Func.Outputs.Count = 1 then
            output
          else
            O.uncombine z |> List.tail |> O.combine
 
        //loop - replace placeholders with generator func output to close the loop
        let out_vars = 
          List.zip out_vars_fwd (O.uncombine newState)
          |> List.map (fun (fw,ac) -> 
            let ac = O.reconcile_dynamic_axis(ac,out_axis)
            ac.Func.ReplacePlaceholders(idict[fw.Var,ac.Var]) |> F )

        match until_predicate with
        | None                  -> output
        | Some until_predicate  ->
            let valid_frames =
                L.Recurrence(initial_states=[Node.Scalar 1.0])                   // initial state
                  (fun (h:Node) (x:Node) -> (1.0 - O.seq_past_value(x,1)) * h) //step function: stops when input returns a 1
                  (until_predicate output)                                     //input: zeros followed by a 1 (in most cases)                      

            let output = O.seq_gather(output, valid_frames, name="valid_output")
            output
        


        
