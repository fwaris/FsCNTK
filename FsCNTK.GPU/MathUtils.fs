namespace FsCNTK
module MathUtils =
    open System

    let epsilon = 0.000001

    let normalize (ls:float[]) = 
      let ls' = ls |> Array.map (fun l -> if l = 0.0 then epsilon else l)
      let s = 1.0 / (ls' |> Array.sum)
      ls'|> Array.map ((*) s)

    let softmax (ls:float[]) = 
      let ls' = ls |> Array.map (fun l -> if l = 0.0 then Math.Exp(epsilon) else Math.Exp (l))
      let s = 1.0 / (ls' |> Array.sum)
      ls'|> Array.map ((*) s)

     //[|1.0; 2.0; 3.0; 4.0; 1.0; 2.0; 3.0|] |> softmax

    let crossEntropyLoss ys y's   = 
        let l = 
            Array.zip ys y's 
            |> Array.map(fun (y,y') -> 
                let y  = if y = 0.0 then epsilon else y
                let y' = if y' = 0.0 then epsilon else y'
                let i = y * log (y/y')
                //printfn "%f * log (%f / %f) = %f " y y y' i
                i)
            |> Array.sum
        //printfn "%A" l
        l

    let sseLoss (ys:float[]) (y's:float[])  = 
      Array.zip  ys  y's
      |> Array.map (fun (y, y') -> (y' - y) ** 2.0 )
      |> Array.sum 

    let private sqr x = x * x

    let stddev fs = 
      let m = fs |> Array.average
      let sse = fs |> Array.map(fun f -> f - m |> sqr) |> Array.sum
      let variance = sse/(float fs.Length)
      sqrt variance


