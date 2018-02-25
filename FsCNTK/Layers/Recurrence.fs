namespace FsCNTK
open CNTK
open System
open FsBase
open Blocks
open Layers

#nowarn "25"

module Layers_Recurrence =
  open FsBase
  
  type private Cell = LSTM | GRU | RNNStep
          
  type StepFunction = Shape list (*shapes of states*) * (Node list (*states*) -> Node (* input *) -> Node list)

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

    static member Stabilizer
      (
        ?steepness,
        ?enable_self_stabilization,
        ?name
      )
      =
        let steepness = defaultArg steepness 4
        let enable_self_stabilization = defaultArg enable_self_stabilization true
        let name = defaultArg name ""

        fun (x:Node) ->
          if not enable_self_stabilization then x
          else
            let init_parm = Math.Log(Math.Exp(float steepness) - 1.0) / (float steepness)
            let param = Node.Parm(Ds[],init=init_parm,name="alpha")
            let param = if steepness = 1 then param else (float steepness) .* param 
            let beta = O.softplus param 
            let r = beta .* x
            r

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

      let Sdh = L.Stabilizer(enable_self_stabilization=enable_self_stabilization, name="dh_stabilizer")
      let Sdc = L.Stabilizer(enable_self_stabilization=enable_self_stabilization, name="dc_stabilizer")
      //let Sct = L.Stabilizer(enable_self_stabilization=enable_self_stabilization, name="c_stabilizer") //for peepholes
      let Sht = L.Stabilizer(enable_self_stabilization=enable_self_stabilization, name="P_stabilizer")

      let b = Node.Parm(               cell_shape_stacked,   init=init_bias, name="b")
      let W = Node.Parm(   O.shape x + cell_shape_stacked,   init=init, name="W")
      let H = Node.Parm(   out_shape + cell_shape_stacked_H, init=init, name="H")

      let H1 = 
        match cellType with 
        | GRU -> Node.Parm(out_shape + cell_shape          , init=init, name="H1") |> Some
        | _   -> None

      {
          stack_axis            = axisVector [new Axis(-1)] 
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
      let lstm (x:Node) (dh,dc)   =
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
        let dcs = rp.Sdc(dc) //stabilized previous cell state

        //projected contribution from inputs(s), hidden and bias
        let prj1 = (x * rp.W) 
        let prj2 = (dhs * rp.H) 
        let proj4 = rp.b + prj1 + prj2

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

        h,c

      let h_shape = out_shape
      let c_shape = defaultArg cell_shape out_shape

      //step_function
      [h_shape; c_shape], fun (h::c::_) (x:Node) -> let (h',c') = lstm x (h,c) in (h'::c'::[])

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

      let gru (x:Node) dh =

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
      [out_shape],fun (h::_) (x:Node) ->  [gru x h]


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

      let rnn_step (x:Node) dh =

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

      [out_shape],fun (h::_) (x:Node) ->  [rnn_step x h]

    static member RecurrenceFrom 
      (
       step_function:StepFunction, 
       ?go_backwards,
       ?name
      )
      =
      let go_backwards = defaultArg  go_backwards false //'false' means get past value - somewhat counter intuitive but matches python
      let name = defaultArg name ""

      let state_shapes,func = step_function

      fun  states (x:Node) ->

        let time_step = if go_backwards then -1 else +1

        if state_shapes.Length <> List.length states then failwith "number of states should match step function"

        let out_forward_vars = 
          List.zip states state_shapes 
          |> List.map (fun (n,s) ->  
            Node.Placeholder
              (
                s, 
                x.Var.DynamicAxes))

        let out_vars = func out_forward_vars x
        let out_actual = List.zip out_vars states |> List.map (fun (v,s) -> L.delay(v,Some s,time_step,""))

        //rewire to make recurrence loop and remove placeholder variables
        let out_vars = 
          List.zip out_forward_vars out_actual
          |> List.map (fun (fw,ac) -> 
            ac.Func.ReplacePlaceholders(idict[fw.Var,ac.Var]) |> F )

        out_vars

    static member Recurrence
      (
       step_function:StepFunction, 
       ?go_backwards,
       ?initial_states,
       ?init_value,     
       ?name
      )                            
      =
      match initial_states,init_value with
      | Some _, Some _ -> failwithf "Recurrence: both inital_states and inital_value should not be specified together"
      | _,_            -> ()

      fun (x:Node) -> 
        let go_backwards = defaultArg  go_backwards false
        let name = defaultArg name ""
        let state_shapes,_ = step_function

        let init_value = defaultArg init_value 0.0
        let initial_states = 
          match initial_states with
          | None -> state_shapes |> List.map (fun s ->  new Constant(!>[|1|], dataType, init_value) :> Variable |> V )
          | Some rs -> rs
      
        let recurrence_from = L.RecurrenceFrom(step_function,go_backwards=go_backwards,name=name)

        if !Layers.trace then printfn ">> Recurrence with %A" (initial_states |> List.map (O.shape>>dims))
        recurrence_from initial_states x