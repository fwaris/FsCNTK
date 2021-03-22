namespace FsCNTK
open CNTK
type C = CNTKLib
//api support for training

type T =

  static member private to_schedule (schedule:(int*float) seq) =
    schedule
    |> Seq.map (fun (n,r) ->  new PairSizeTDouble(uint32 n,r))
    |> ResizeArray
    |> fun x -> new VectorPairSizeTDouble(x)

  static member schedule(schedule : #seq<(int*float)> , minibatch_size, epoch_size) =
    new TrainingParameterScheduleDouble(T.to_schedule schedule, uint32 epoch_size, uint32 minibatch_size)

  static member schedule(schedule : float, minibatch_size) =
    new TrainingParameterScheduleDouble(schedule, uint32 minibatch_size)

  static member schedule_per_sample(schedule:#seq<float>) = 
    let schedule = schedule |> Seq.map (fun x->1,x) |> T.to_schedule
    new TrainingParameterScheduleDouble(schedule)

  static member schedule_per_sample(schedule:#seq<float>, epoch_size) = 
    let schedule = schedule |> Seq.map (fun x->1,x) |> T.to_schedule
    new TrainingParameterScheduleDouble(schedule, uint32 epoch_size)

  static member schedule_per_sample(schedule:seq<int*float>, epoch_size) = 
    let schedule = schedule |> T.to_schedule
    new TrainingParameterScheduleDouble(schedule, uint32 epoch_size)

  static member schedule_per_sample(schedule:float) = 
    new TrainingParameterScheduleDouble(schedule, uint32 1)

  static member schedule_per_sample(schedule:float, epoch_size) = 
    new TrainingParameterScheduleDouble(T.to_schedule [0,schedule], uint32 epoch_size,  1u)

  static member momentum_schedule (x:float) = T.schedule_per_sample x

  static member momentum_schedule (x:float, minibatch_size) =
    new TrainingParameterScheduleDouble(x, uint32 minibatch_size)