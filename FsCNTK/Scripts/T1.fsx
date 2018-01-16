﻿#load "SetEnv.fsx"
open FsCNTK
open FsCNTK.FsBase
open FsCNTK.Layers
open Layers_Dense
open Layers_BN
open Layers_ConvolutionTranspose2D
open Layers_ConvolutionTranspose
open Layers_Convolution2D
open CNTK
open System.IO

//scratch pad

type C = CNTKLib

let imageSize = 28 * 28
let numClasses = 10

let img_h, img_w = 28, 28
let kernel_h, kernel_w = 5, 5
let stride_h, stride_w = 2, 2
let g_input_dim = 100
let g_output_dim = img_h * img_w
let d_input_dim = g_output_dim
let s_h2, s_w2 = img_h / 2, img_w / 2 //Input shape (14,14)
let s_h4, s_w4 = img_h / 4, img_w / 4 //Input shape (7,7)
let gfc_dim = 1024
let gf_dim = 64

let gkernel,dkernel =
    if kernel_h = kernel_w then
        kernel_h,kernel_h
    else
        failwith "This tutorial needs square shaped kernel"

let gstride,dstride =
    if stride_h = stride_w then
       stride_h, stride_h
    else
        failwith "This tutorial needs same stride in all dims"

let bn_with_relu  = 
  L.BN (map_rank=1) 
  >> L.Activation Activation.ReLU

let d2 =
  L.Dense (D gfc_dim, name="h0")
  >> bn_with_relu
  >> L.Dense (Ds [gf_dim *2; s_h4; s_w4], name="h1")
  >> bn_with_relu
  >> L.ConvolutionTranspose2D
      (
        D gkernel,
        num_filters=gf_dim*2,
        strides=D gstride,
        pad=true, output_shape=Ds[s_h2; s_w2]
      )

let tm()=
  let Z = Node.CreateVar(D g_input_dim,dynamicAxes=[Axis.DefaultBatchAxis()])
  let dout = d2 Z
  shape dout

let tct()=
  let f = L.ConvolutionTranspose(Ds [3;4], 128, output_shape=Ds[482;643],pad=false)
  let Z = Node.CreateVar(Ds [3;480;640],dynamicAxes=[Axis.DefaultBatchAxis()])
  let m = f Z
  shape m
  ()

let tctb()=
  let Z = Node.CreateVar(Ds [3;480;640],dynamicAxes=[Axis.DefaultBatchAxis()])
  let p = new Parameter(!--( Ds[3;128;3;4]), dataType,0.,device,"w")
  let osp = Ds[128;482;643]
  let f = C.ConvolutionTranspose(p,Z.Var,!-- (D 1),boolVector[true],boolVector[false],!--osp)
  let fn = F f 
  shape fn

  ()

let t1() =
  let x = Node.CreateVar(Ds [128;7;7], dynamicAxes=[Axis.DefaultBatchAxis()])
  let W = Node.CreateParm(Ds [128;5;5])
  let out = Ds [128;14;14]
  let st  = D 1
  let s = boolVector [true]
  let p = boolVector [true]
  let c = C.ConvolutionTranspose(W.Var,x.Var,!--st,s,p,!--out) |> F
  shape c

let t2() =
  let x = Node.CreateVar(Ds [1024], dynamicAxes=[Axis.DefaultBatchAxis()])
  let w = Node.CreateParm(Ds [1024; 128;7;7])

  w.Var.Shape.Dimensions
  x.Var.Shape.Dimensions

  let p = C.Times(w.Var,x.Var, 3u, 0) |> F
  shape p
