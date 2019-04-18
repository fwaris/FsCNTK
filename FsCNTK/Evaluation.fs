namespace FsCNTK
open CNTK
open FsBase
type C = CNTKLib

// F# wrappers for model evaluation and value conversion

type E =

  static member eval args = 
    fun (n:Node) ->
      let out = n.Func.Outputs |> Seq.map(fun o->o,(null:Value)) |> idict
      n.Func.Evaluate(args,out,device)
      out

  ///function takes only one argument 
  static member eval(mb:MinibatchData) =
    fun (n:Node) ->
      let argV = n.Func.Arguments.[0] //expect only 1 argument
      E.eval(idict[argV,mb.data]) n
  
  ///from streamed data
  static member eval(mbs:UnorderedMapStreamInformationMinibatchData) =
    fun (strmInf:StreamInformation seq) (n:Node) ->
      let args = Seq.zip n.Func.Arguments strmInf |> Seq.map(fun (v,s)->v,mbs.[s].data) |> idict
      E.eval args n  
      
  //eval single output
  static member eval1 args = 
    fun (n:Node) ->
      let outv = n.Func.Output
      let outp = idict [outv,(null:Value)] 
      n.Func.Evaluate(args,outp,device)
      outp.[outv] |> Vl.getArray

  static member weights (n:Node) =
    n.Func.Parameters() 
    |> Seq.map(fun p -> 
        let v2 = Value.Create(p.Shape,[p.Value()],device)
        p.Name, p.Shape.Dimensions |>Seq.toList, v2 |> Vl.getArray)
    |> Seq.toArray