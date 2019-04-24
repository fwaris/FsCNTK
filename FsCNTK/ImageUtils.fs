module ImageUtils
open System.Windows.Forms
open System.Drawing
open System.Runtime.InteropServices
open System.Drawing.Imaging
open System

let private scaler (sMin,sMax) (vMin,vMax) (v:float) =
    if v < vMin then failwith "out of min range for scaling"
    if v > vMax then failwith "out of max range for scaling"
    (v - vMin) / (vMax - vMin) * (sMax - sMin) + sMin
    (*
    scaler (0.1, 0.9) (10., 500.) 223.
    scaler (0.1, 0.9) (10., 500.) 10.
    scaler (0.1, 0.9) (10., 500.) 500.
    scaler (0.1, 0.9) (-200., -100.) -110.
    *)

let toGray (w,h) (vals:byte[]) =
    let xs = vals |> Seq.collect (fun g ->  [g; g; g; 255uy])|> Seq.toArray
    let bmp = new Bitmap(w,h,Imaging.PixelFormat.Format32bppArgb)
    let data = bmp.LockBits(
                    new Rectangle(0, 0, bmp.Width, bmp.Height),
                    ImageLockMode.ReadWrite,
                    bmp.PixelFormat);
    Marshal.Copy(xs, 0, data.Scan0, xs.Length);
    bmp.UnlockBits(data)
    bmp
    
let show (bmp:Bitmap) = 
    let form = new Form()
    form.Width  <- 400
    form.Height <- 400
    form.Visible <- true 
    form.Text <- "Image"
    let p = new PictureBox(
                    Image=bmp,
                    Dock = DockStyle.Fill,
                    SizeMode=PictureBoxSizeMode.StretchImage)
    form.Controls.Add(p)
    form.Show()

let showGrid title (w,h) imgList =
    let form = new Form()
    form.Width  <- 600
    form.Height <- 400
    form.Visible <- true 
    form.Text <- title
    let grid = new TableLayoutPanel()
    grid.AutoSize <- true
    grid.ColumnCount <- w
    let cpct = 100.f / float32 w
    for _ in 1..w do
        grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, cpct)) |> ignore
    grid.RowCount <- h
    let rpct = 100.f / float32 h
    for _ in 1 .. h do
        grid.RowStyles.Add(new RowStyle(SizeType.Percent,rpct)) |> ignore
    grid.GrowStyle <-  TableLayoutPanelGrowStyle.AddRows
    grid.Dock <- DockStyle.Fill
    imgList |> Seq.iter (fun bmp -> 
        let p = new PictureBox(
                    Image=bmp,
                    Dock = DockStyle.Fill,
                    SizeMode=PictureBoxSizeMode.StretchImage)
        grid.Controls.Add p)
    form.Controls.Add(grid)
    form.Show()
