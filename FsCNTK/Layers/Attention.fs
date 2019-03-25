namespace FsCNTK
open CNTK
open FsCNTK
open System
open FsBase
open Blocks
open Layers_Dense
open Layers_Sequence

module Models_Attention =
  type M =

    //Layer factory function to create a function object that implements an attention model
    //as described in Bahdanau, et al., "Neural machine translation by jointly learning to align and translate."
    static member AttentionModel 
        (
            attention_dim : Shape, 
            ?attention_span,
            ?attention_axis,
            ?init,
            ?go_backwards,
            ?enable_self_stabilization,
            ?name
        ) = 

        let init                      = defaultArg init (C.GlorotUniformInitializer())
        let go_backwards              = defaultArg go_backwards false
        let enable_self_stabilization = defaultArg enable_self_stabilization true

        let compatible_attention_mode =
          match attention_span, attention_axis with
          | None  , Some _         -> failwith "attention_axis cannot be None when attention_span is not None"
          | None  , _              -> false
          | Some _, None           -> failwith "attention_span cannot be None when attention_axis is not None"
          | Some a, _  when a <= 0 -> failwith "attention_span must be a positive value"
          | _, _                   -> true

        //model parameters
        let attn_proj_enc   = B.Stabilizer(enable_self_stabilization=enable_self_stabilization) // projects input hidden state, keeping span axes intact
                            >> L.Dense(attention_dim, init=init, input_rank=1, bias=false) 

        let attn_proj_dec   = B.Stabilizer(enable_self_stabilization=enable_self_stabilization) // projects decoder hidden state, but keeping span and beam-search axes intact
                            >> L.Dense(attention_dim, init=init, input_rank=1, bias=false) 

        let attn_proj_tanh  = B.Stabilizer(enable_self_stabilization=enable_self_stabilization) // projects tanh output, keeping span and beam-search axes intact
                            >> L.Dense(D 1          , init=init, input_rank=1, bias=false) 

        let attn_final_stab = B.Stabilizer(enable_self_stabilization=enable_self_stabilization)

        let new_attention (encoder_hidden_state:Node, decoder_hidden_state:Node) =
            let unpacked_ehs = O.seq_unpack(encoder_hidden_state, padding_value=0.0)
            let unpacked_encoder_hidden_state, valid_mask = O.getOutput 0 unpacked_ehs, O.getOutput 1 unpacked_ehs
            let projected_encoder_hidden_state = O.seq_broadcast_as(attn_proj_enc(unpacked_encoder_hidden_state), decoder_hidden_state)
            let broadcast_valid_mask = O.seq_broadcast_as(O.reshape(valid_mask, Ds[1], new Axis(1)), decoder_hidden_state)
            //[#,d] [1,*]
            let projected_decoder_hidden_state = attn_proj_dec(decoder_hidden_state)
            let tanh_output = O.tanh(projected_decoder_hidden_state + projected_encoder_hidden_state)
            let attention_logits = attn_proj_tanh(tanh_output)
            //let minus_inf = Node.Scalar -1e+30 
            let minus_inf = new Constant(!> [| |], dataType, -1e+30 ) :> Variable |> V 
            let masked_attention_logits = O.element_select(broadcast_valid_mask, attention_logits, minus_inf)
            let attention_weights = O.softmax(masked_attention_logits, axis=new Axis(0))
            let attention_weights = L.Label("attention_weights")(attention_weights)

            let attended_encoder_hidden_state = 
                O.reduce_sum(
                    attention_weights .* O.seq_broadcast_as(unpacked_encoder_hidden_state, attention_weights), 
                    axis=new Axis(0))

            let output = attn_final_stab(O.reshape(attended_encoder_hidden_state, Ds[], new Axis(0), new Axis(1)))
            if !Layers.trace then printfn ">> New Attension Model with %A" (dims attention_dim)
            output

        let old_attention (h_enc:Node, h_dec:Node) = 
            let history_axis = h_dec //used for passing axis information
            let attention_span = match attention_span with Some i -> i | None -> failwithf "attention_span is required"
            let attention_axis = match attention_axis with Some s -> s | None ->  failwith "attenion_axis is required"
            //window over encoder with associated mask
            let pv =  L.PastValueWindow(attention_span,attention_axis,go_backwards) h_enc
            let pvs = O.uncombine pv
            let h_enc, (*mask*) h_enc_valid = pvs.[0], pvs.[1]
            let h_enc_proj = attn_proj_enc h_enc
            //broadcast window to every decoder timestep
            let h_enc_proj = O.seq_broadcast_as(h_enc_proj, history_axis)
            let h_enc_valid = O.seq_broadcast_as(h_enc_valid, history_axis)
            // --- decoder state
            //project decoder hidden state
            let h_dec_proj = attn_proj_dec h_dec
            let tanh_out = O.tanh(h_dec_proj + h_enc_proj) // (attension_span, attention_dim)
            let u = attn_proj_tanh(tanh_out)               // (attention_span, 1)
            let u_masked = u + ((h_enc_valid - 1.) .* 50.) // logzero-out the unused elements for the softmax denominator. 50. is somewhat arbitrary
            let attention_weights = O.softmax(u_masked,axis=attention_axis)
            let attention_weights = L.Label("attention_weights") attention_weights
            //now take the weighte`d sum over the encoder state vectors
            let h_enc = O.seq_broadcast_as(h_enc, history_axis)
            let h_weighted = h_enc .* attention_weights
            let h_att = O.reduce_sum(h_weighted, axis=attention_axis)
            if !Layers.trace then printfn ">> Old Attension Model with %A" (dims attention_dim)
            h_att

        if compatible_attention_mode then   
            old_attention
        else
            new_attention

