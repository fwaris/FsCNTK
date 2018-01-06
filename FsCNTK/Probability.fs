module Probability
open System
open System.Threading

//xorshift128plus implementation, https://en.wikipedia.org/wiki/Xorshift
type XorshiftPRNG(seed) =
    let mutable s : uint64[] = Array.zeroCreate 2

    do s.[1] <- uint64 seed

    let sample() =
        let mutable x = s.[0]
        let y = s.[1]
        s.[0] <- y
        x <- x ^^^ (x <<< 23)
        s.[1] <- x ^^^ y ^^^ (x >>> 17) ^^^ (y >>> 26)
        let smpl = s.[1] + y
        if smpl = System.UInt64.MaxValue then smpl - 1UL else smpl

    member x.NextDouble() = (float (sample())) / float System.UInt64.MaxValue

    member x.Next(max) = 
        if max < 0 then failwith "max < 0"
        x.NextDouble() * (float max) |> int

    member x.Next(min:int,max:int) = 
        if min > max then failwith "min > max" 
        let r = max - min in (float r) * (x.NextDouble()) + (float min) |> int

    new()=XorshiftPRNG(System.Environment.TickCount)

//thread-safe random number generator
//let RNG =
//    // Create master seed generator and thread local value
//    let seedGenerator = new Random();
//    let localGenerator = new ThreadLocal<Random>(fun _ -> 
//        lock seedGenerator (fun _ -> 
//        let seed = seedGenerator.Next()
//        new Random(seed)))
//    localGenerator

let RNG =
    // Create master seed generator and thread local value
    let seedGenerator = new Random()
    let localGenerator = new ThreadLocal<XorshiftPRNG>(fun _ -> 
        lock seedGenerator (fun _ -> 
        let seed = seedGenerator.Next()
        new XorshiftPRNG(seed)))
    localGenerator

//Marsaglia polar method
let private gaussian() = 
    let mutable v1 = 0.
    let mutable v2 = 0.
    let mutable s = 2.
    while s >= 1. || s = 0. do
        v1 <- 2. * RNG.Value.NextDouble() - 1.
        v2 <- 2. * RNG.Value.NextDouble() - 1.
        s <- v1 * v1 + v2 * v2
    let polar = sqrt(-2.*log(s) / s)
    polar,v1,v2

let private spare = new ThreadLocal<_>(fun () -> ref None)

//thread-safe Z sampler
let ZSample() = 
    match spare.Value.Value with
    | None -> 
        let polar,v1,v2 = gaussian()
        spare.Value := Some (polar,v2)
        v1*polar
    | Some(polar,v2) -> 
        spare.Value := None
        v2*polar

//thread-safe gaussian sampler
let GAUSS mean sigma = 
    match spare.Value.Value with
    | None -> 
        let polar,v1,v2 = gaussian()
        spare.Value := Some (polar,v2)
        v1*polar*sigma + mean
    | Some(polar,v2) -> 
        spare.Value := None
        v2*polar*sigma + mean

let createWheel (weights:('a*float)[]) = //key * weight  key must be unique
    let s = Array.sumBy snd weights
    if s = 0. then failwithf "weights cannot sum to 0 %A" s
    let ws = 
        weights 
        |> Array.filter (fun (_,w) -> w > 0. && w <> nan) 
        |> Array.map (fun (k,w) -> k, w / s)        //total sums to 1 now
        |> Array.sortBy snd                         //arrange ascending
    let cum = 
        if weights.Length <= 1 then
            ws
        else
            (ws.[0],ws.[1..])||>Array.scan (fun (_,acc) (k,w) -> k,acc + w)
    ws,cum

let spinWheel wheel = 
    let r = RNG.Value.NextDouble()
    wheel |> Array.pick(fun (k,w) -> if w > r then Some k else None)

   
(*
#load "Probability.fs"
open Probability
let reqs =  [for i in 1 .. 1000 -> async{return (GAUSS 10. 3.)}]
let rs = Async.Parallel reqs |> Async.RunSynchronously
let reqs2 =  [for i in 1 .. 1000 -> async{return ZSample()}]
let rs2 = Async.Parallel reqs2 |> Async.RunSynchronously

let m,w = createWheel [|'a',2.; 'b',1.|]
[for i in 0..100 ->(spinWheel w),1] |> List.groupBy fst |> List.map (fun (x,ys)->x,List.sumBy snd ys)

let rng = System.Random()// 
let rng = XorshiftPRNG()
for i in 0 .. 1000 do printfn "%A" (rng.Next())
for i in 0 .. 1000000 do if rng.Next(0,10) >= 10 then printfn "10+"
for i in 0 .. 1000000 do if rng.Next(0,10) >= 9 then printfn "9+"
*)
