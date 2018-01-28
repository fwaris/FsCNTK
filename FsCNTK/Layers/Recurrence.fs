namespace FsCNTK
open CNTK
open System
open FsBase
open Blocks
open Layers

module Layers_Recurrence =

  type Cell = LSTM | GRU | RNNStep

  type L with

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
            let param = new Parameter(!-- Ds[], dataType, init_parm, device, "alpha")
            let param = if steepness = 1 then !> param else !> C.ElementTimes((scalar (float steepness)), param)
            let beta = C.Softplus(param)
            let r = C.ElementTimes(!> beta, x.Var)
            r |> F


    static member RecurrentBlock
      (
        cellType,
        out_shape,
        ?cell_shape,
        ?activation,
        ?init,
        ?init_bias,
        //?use_peephole, - not that useful according to "LSTM: A Search Space Odyssey", https://arxiv.org/pdf/1503.04069.pdf
        ?enable_self_stabilization,
        ?name
      )
      =
      let activation = defaultArg activation Activation.Tanh
      let init = defaultArg init (C.GlorotUniformInitializer())
      let init_bias = defaultArg init_bias 0.0
      let enable_self_stabilization = defaultArg enable_self_stabilization false
      let name = defaultArg name ""
      
      let has_projection,cell_shape = 
        match cell_shape with 
        | None | Some Shape.Unknown -> false,out_shape 
        | Some c                    -> true,c
      
      if len out_shape <> 1 || len cell_shape <> 1 then
        failwithf "LSTM shape and cell_shape must be vectors"

      //see python code and comment in blocks.py
      //for stacking of multiple variables along the fastest changing
      //dimension for efficient communication 
      let stacked_dim = cell_shape |> dims |> List.last // or first as only 1 dimension

      let stacked_dim = stacked_dim * match cellType with 
                                      | RNNStep ->  1 
                                      | GRU     ->  3 //3 gates
                                      | LSTM    ->  4 //4 gates

      let cell_shape_stacked = D stacked_dim

      let stacked_dim_h = stacked_dim * match cellType with 
                                        | RNNStep ->  1 
                                        | GRU     ->  2 
                                        | LSTM    ->  4

      let cell_shape_stacked_H  = D stacked_dim_h

      fun (x:Node) ->
        let b = new Parameter(!--             cell_shape_stacked   , init_bias, device, "b")
        let W = new Parameter(!--(O.shape x +   cell_shape_stacked)  , dataType, init, device, "W")
        let H = new Parameter(!--(out_shape + cell_shape_stacked_H), dataType, init, device, "H")

        let H1 = 
          match cellType with 
          | GRU -> new Parameter(!--(out_shape + cell_shape), dataType, init, device, "H1")
          | _   -> null
        
        let Wmr = 
          if has_projection 
          then new Parameter(!-- (cell_shape + out_shape), dataType, init, device, "P") 
          else null
        
        let Sdh = L.Stabilizer(enable_self_stabilization=enable_self_stabilization, name="dh_stabilizer")
        let Sdc = L.Stabilizer(enable_self_stabilization=enable_self_stabilization, name="dc_stabilizer")
        let Sct = L.Stabilizer(enable_self_stabilization=enable_self_stabilization, name="c_stabilizer")
        let Sht = L.Stabilizer(enable_self_stabilization=enable_self_stabilization, name="P_stabilizer")

        let stack_axis = axisVector [new Axis(-1)] 

        let applyActvn v = (L.Activation activation (F v)).Var //utility funciton to apply activation 

        //Great reference: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        let lstm (x:Node) dh dc   =
          let dhs = Sdh(dh) //stabilized previous output 
          let dcs = Sdc(dc) //stabilized previous cell state

          //projected contribution from inputs(s), hidden and bias
          //Q: is this multiplication order correct for .Net API
          let proj4 = C.Plus(b, !>C.Plus(!> C.Times(W, x.Var), !> C.Times(H, dhs.Var)))
          
          let it_proj  = C.Slice(!>proj4, stack_axis, intVector [0*stacked_dim], intVector [1*stacked_dim])  // split along stack_axis
          let bit_proj = C.Slice(!>proj4, stack_axis, intVector [1*stacked_dim], intVector [2*stacked_dim])
          let ft_proj  = C.Slice(!>proj4, stack_axis, intVector [2*stacked_dim], intVector [3*stacked_dim])
          let ot_proj  = C.Slice(!>proj4, stack_axis, intVector [3*stacked_dim], intVector [4*stacked_dim])

          let it  = C.Sigmoid(!>it_proj)                //input gate(t)
          let bit = C.Times(!>it,applyActvn bit_proj)   //applied to tanh of input network
          let ft  = C.Sigmoid(!>ft_proj)                //forget-me-not gate(t)
          let bft = C.ElementTimes(!>ft,dc.Var)             //applied to cell(t-1)
          let ct  = C.Plus(!>bft, !>bit)                  //c(t) is sum of both
          let ot  = C.Sigmoid(!>ot_proj)                //output gate(t)
          let ht  = C.ElementTimes(!>ot,applyActvn ct)  //applied to tanh(cell(t))

          let c = ct                                  //cell value
          let h = if has_projection then C.Times(Wmr, !>Sht(F ht)) else ht

          F h , F c

        let gru (x:Node) dh =

          let dhs = Sdh(dh) //previous value stabilized

          let projx3 = C.Plus(b, !> C.Times( W, x.Var))
          let projh2 = C.Times(H, dhs.Var)

          let zt_proj = C.Plus
                          (
                            !> (C.Slice (!>projx3, stack_axis, intVector [0*stacked_dim], intVector [1*stacked_dim])),
                            !> (C.Slice (!>projh2, stack_axis, intVector [0*stacked_dim], intVector [1*stacked_dim]))
                          )
          let rt_proj = C.Plus
                          (
                            !>(C.Slice (!>projx3, stack_axis, intVector [1*stacked_dim], intVector [2*stacked_dim])),
                            !>(C.Slice (!>projh2, stack_axis, intVector [1*stacked_dim], intVector [2*stacked_dim]))
                          )
          let ct_proj = C.Slice (!>projx3, stack_axis, intVector [2*stacked_dim], intVector [3*stacked_dim])

          let zt = C.Sigmoid (!>zt_proj)        // update gate z(t)
          let rt = C.Sigmoid (!>rt_proj)        // reset gate r(t)
          let rs = C.ElementTimes(dhs.Var,!>rt)     // "cell" c
          let ct = applyActvn (C.Plus(!>ct_proj, !>C.Times(H1, !>rs)))

          //Python:  ht = (1 - zt) * ct + zt * dhs
          let ht = C.Plus(!>C.ElementTimes(!>C.Minus(scalar 1., !>zt),ct),!>C.ElementTimes(!>zt, !>dhs)) // hidden state ht / output

          let h = if has_projection then C.Times(Wmr,Sht(F ht).Var) else ht

          F h

        let rnn_step (x:Node) dh  =
          let dhs = Sdh(dh)
          let ht = applyActvn (C.Plus(b,!>C.Plus(!>C.Times(W,x.Var),!>C.Times(H,dhs.Var))))
          let h  = if has_projection then !> (C.Times(Wmr,!>Sht(V ht))) else ht
          h

        match cellType with
        | LSTM    -> lstm x
        | GRU     -> gru x
        | RNNStep -> rnn_step x

    static member Recurrence
        (
            ?dropout_rate,
            ?keep_prob,
            ?seed,
            ?name
        )                            
        =
        let drop_rate = 
          match dropout_rate,keep_prob with
          | None  , None   -> failwith "Dropout: either dropout_rate or keep_prob must be specified."
          | Some _, Some _ -> failwith "Dropout: dropout_rate and keep_prob cannot be specified at the same time."
          | _     , Some k when  k < 0.0 || k >= 1.0 -> failwith "Dropout: keep_prob must be in the interval [0,1)"
          | _     , Some k -> 1.0 - k
          | Some d, _      -> d

        fun (x:Node) ->
          let r =
            match seed with
            | Some s ->  C.Dropout(x.Var, drop_rate, uint32 s)
            | None   ->  C.Dropout(x.Var ,drop_rate)

          F r
      
