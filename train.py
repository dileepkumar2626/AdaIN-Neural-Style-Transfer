import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from utils.utils import *
from utils.models import *
from tqdm import tqdm
from PIL import Image,ImageFile
from torchvision.utils import save_image
ImageFile.LOAD_TRUNCATED_IMAGES = True
def parse_arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument("--content_dir",type=str,default="C:/Users/Dileep Kumar/Desktop/AdaIN-Neural-Style-Transfer/data/test2017",help="Location of content Dataset")
    parser.add_argument("--style_dir",type=str,default="C:/Users/Dileep Kumar/Desktop/AdaIN-Neural-Style-Transfer/data/train_1",help="Location of content dataset")
    parser.add_argument("--vgg",type=str,default="C:/Users/Dileep Kumar/Desktop/AdaIN-Neural-Style-Transfer/vgg_normalised.pth",help="Location of pre-trained VGG")
    parser.add_argument("--experiment",type=str,default="experiment1",help="Name of experiment")
    parser.add_argument("--final_size",type=int,default=256,help="Size of final image")
    parser.add_argument("--content_size",type=int,default=512,help="Size of content image")
    parser.add_argument("--style_size",type=int,default=512,help="Size of style image ")
    parser.add_argument("--crop",action=argparse.BooleanOptionalAction,default=True,help="Random-crop images (use --no-crop to disable)")
    parser.add_argument("--batch_size",type=int ,default=16,help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4,help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5,help='Learning rate decay')
    parser.add_argument('--epochs', type=int, default=200,help='Number of epochs')
    parser.add_argument('--content_weight', type=float, default=1.0,help='Content weight')
    parser.add_argument('--style_weight', type=float, default=5,help='Style weight')
    parser.add_argument('--log_interval', type=int, default=1,help='Log interval')
    parser.add_argument('--save_interval', type=int, default=2,help='Save interval')
    parser.add_argument('--num_workers', type=int, default=min(4, os.cpu_count() or 1),
                        help='DataLoader worker processes (0 = load in the main process, very slow)')
    parser.add_argument('--log_steps', type=int, default=20,
                        help='Refresh the progress-bar losses every N steps (each refresh syncs the GPU)')
    parser.add_argument('--resume', action='store_true', default=False,help='Resume training')
    parser.add_argument('--decoder_path', type=str, default=None,help='Path to decoder checkpoint')
    parser.add_argument('--optimizer_path', type=str, default=None,help='Path to optimizer checkpoint')
    return parser.parse_args()
def main():
    args=parse_arguments()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True   # input size is fixed -> faster conv kernels
    else:
        print("WARNING: CUDA is not available, training on the CPU (this is 20-50x slower). "
              "Install the CUDA build of PyTorch: https://pytorch.org/get-started/locally/")
    save_dir=Path("experiment")/args.experiment
    save_dir.mkdir(exist_ok=True,parents=True)
    with open(save_dir/"args.txt","w") as args_file:
        for key,value in vars(args).items():
            args_file.write(f"{key}: {value}\n")
    content_transform=get_transform(args.content_size,args.crop,args.final_size)
    style_transform=get_transform(args.style_size,args.crop,args.final_size)
    content_dataset=ImageFolderdataset(args.content_dir,content_transform,draft_size=args.content_size)
    style_dataset=ImageFolderdataset(args.style_dir,style_transform,draft_size=args.style_size)
    # Loading + resizing JPEGs is the real bottleneck. With num_workers=0 (the old
    # default) the GPU sat idle while the main process decoded every image.
    loader_kwargs=dict(
        batch_size=args.batch_size,shuffle=True,drop_last=True,
        num_workers=args.num_workers,pin_memory=(device.type=="cuda"),
    )
    if args.num_workers>0:
        loader_kwargs.update(persistent_workers=True,prefetch_factor=4)
    content_dataloader=DataLoader(content_dataset,**loader_kwargs)
    style_datalaoder=DataLoader(style_dataset,**loader_kwargs)
    print("Number of batches in the content dataset: ",len(content_dataloader))
    print("Number of batches in the style dataset: ",len(style_datalaoder))
    encoder=VGGEncoder(args.vgg).to(device)
    decoder=Decoder().to(device)
    optimizer=optim.Adam(decoder.parameters(),lr=args.lr)
    scheduler=optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch:1.0/(1.0+args.lr_decay*epoch))
    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path,map_location=device))
        optimizer.load_state_dict(torch.load(args.optimizer_path,map_location=device))
    print("Training...")
    mse_loss=torch.nn.MSELoss()
    encoder.eval()
    steps_per_epoch=min(len(content_dataloader),len(style_datalaoder))
    for epoch in range(args.epochs):
        progress_bar=tqdm(zip(content_dataloader,style_datalaoder),total=steps_per_epoch)
        # Accumulate on the GPU: calling .item() every step forces a GPU sync
        # (the old loop did it 6 times per step) and stalls the pipeline.
        running=torch.zeros(3,device=device)
        for step,(content_batch,style_batch) in enumerate(progress_bar):
            content_batch=content_batch.to(device,non_blocking=True)
            style_batch=style_batch.to(device,non_blocking=True)
            with torch.no_grad():                      # encoder is frozen: build no graph for it
                c_feats=encoder(content_batch)
                s_feats=encoder(style_batch)
                t=adaptive_instance_normalizaiton(c_feats[-1],s_feats[-1])
            g=decoder(t)
            g_feats=encoder(g)                         # gradients must flow through this one
            loss_c=mse_loss(g_feats[-1],t)*args.content_weight
            loss_s=0
            for g_f,s_f in zip(g_feats,s_feats):
                g_mean,g_std=cal_mean_std(g_f)
                s_mean,s_std=cal_mean_std(s_f)
                loss_s+=mse_loss(g_mean,s_mean)+mse_loss(g_std,s_std)
            loss_s=loss_s*args.style_weight
            loss=loss_c+loss_s
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses=torch.stack([loss.detach(),loss_c.detach(),loss_s.detach()])
            running+=losses
            if step%args.log_steps==0:
                l,lc,ls=losses.tolist()                # single sync every log_steps
                progress_bar.set_description(f'Loss:{l:4f}, Content Loss: {lc:4f}, Style Loss: {ls:4f}')
        scheduler.step()
        running_loss,running_closs,running_sloss=(running/steps_per_epoch).tolist()   # per-step average
        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Iter {epoch+1}: Loss:{running_loss:4f}, Content Loss: {running_closs:4f}, Style Loss: {running_sloss:4f}')
        if (epoch+1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f'decoder_{epoch+1}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_{epoch+1}.pth')
            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g.detach()], dim=0)
                save_image(output, save_dir / f'output_{epoch+1}.png', nrow=args.batch_size)
if __name__ == '__main__':
    main()