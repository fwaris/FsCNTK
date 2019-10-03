module ImageUtils
open System.Windows.Forms
open System.Drawing
open System.Runtime.InteropServices
open System.Drawing.Imaging
open System
open System.Threading.Tasks
open System.Drawing.Drawing2D

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
    
let show t (bmp:Bitmap) = 
    let form = new Form()
    form.Width  <- 400
    form.Height <- 400
    form.Visible <- true 
    form.Text <- t
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

let resize (image:Image) width height =
    let destRect = new Rectangle(0, 0, width, height);
    let destImage = new Bitmap(width, height);
    destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);
    use graphics = Graphics.FromImage(destImage)
    graphics.CompositingMode <- CompositingMode.SourceCopy;
    graphics.CompositingQuality <- CompositingQuality.HighQuality;
    graphics.InterpolationMode <- InterpolationMode.HighQualityBicubic;
    graphics.SmoothingMode <- SmoothingMode.HighQuality;
    graphics.PixelOffsetMode <- PixelOffsetMode.HighQuality;
    use wrapMode = new ImageAttributes()
    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
    graphics.DrawImage(image, destRect, 0, 0, image.Width,image.Height, GraphicsUnit.Pixel, wrapMode);
    destImage


let getPixelMapper(pixelFormat, heightStride) =
    match pixelFormat with
    | PixelFormat.Format32bppArgb -> (fun (h, w, c) -> h * heightStride + w * 4 + c) 
    | PixelFormat.Format24bppRgb ->  (fun (h, w, c) -> h * heightStride + w * 3 + c)
    | x -> failwithf "unsupported pixel format %A" x

let parallelExtractCHW (image:Bitmap) =
    let channelStride = image.Width * image.Height
    let imageWidth = image.Width
    let imageHeight = image.Height

    let features:byte[] = Array.zeroCreate (imageWidth * imageHeight * 3)
    let bitmapData = image.LockBits(new System.Drawing.Rectangle(0, 0, imageWidth, imageHeight), ImageLockMode.ReadOnly, image.PixelFormat)
    let ptr = bitmapData.Scan0;
    let bytes = Math.Abs(bitmapData.Stride) * bitmapData.Height;
    let rgbValues:byte[] = Array.zeroCreate bytes

    let stride = bitmapData.Stride;

    // Copy the RGB values into the array.
    System.Runtime.InteropServices.Marshal.Copy(ptr, rgbValues, 0, bytes)

    // The mapping depends on the pixel format
    // The mapPixel lambda will return the right color channel for the desired pixel
    let mapPixel = getPixelMapper(image.PixelFormat, stride);

    Parallel.For(0, imageHeight, (fun h ->
        Parallel.For(0, imageWidth, (fun w ->
            Parallel.For(0, 3, (fun c ->
              features.[channelStride * c + imageWidth * h + w] <- rgbValues.[mapPixel(h, w, c)]
            ))  |> ignore
        )) |> ignore
    )) |> ignore

    image.UnlockBits(bitmapData)

    features |> Array.map float32
