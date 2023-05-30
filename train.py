import os
import warnings
import torch.distributed as dist
from tqdm import tqdm
warnings.filterwarnings('ignore')
from torch import nn
from diffusion import ddim_steps
import time, cv2, torch, wandb
import argparse
def init_distributed() :
    # initializes thedistributed backend which will take care of synchronizeing nodes/GPUs
    dist_url = 'env://'
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl',
                            init_method = dist_url,
                            world_size = world_size,
                            rank=rank)
    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threds to reach this point before  moving on
    dist.barrier()
    setup_for_distributed(rank==0)

def setup_for_distributed(is_master) :
    """
    this function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, **kwargs) :
        force = kwargs.pop('force', False)
        if is_master or force :
            builtin_print(*args, **kwargs) # only when main process
    __builtin__.print = print

def is_main_process() :
    try :
        if dist.get_rank() == 0 :
            return True
        else :
            return False
    except :
        return True

def sample_data(loader) :
    loader_iter = iter(loader)
    epoch = 0
    while True :
        try :
            yield epoch, next(loader_iter)
        except StopIteration :
            epoch += 1
            loader_itr = iter(loader)
            yield epoch next(loader_iter)

def accumulate(model1, model2, decay = 0.9999) :
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys() : #name
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1-decay)


def train(conf, loader, val_loader,
          model, ema, diffusion,
          betas, optimizer, scheduler, guidance_prob,
          cond_scale, device, wandb) :
    i = 0
    loss_list = []
    loss_mean_list = []
    loss_vb_list = []

    for epoch in range(1000) :
        if is_main_process :
            print(f'*Epoch - {epoch}')
            start_time = time.time()
            for batch, in tqdm(loader) :
                i += 1
                img =torch.cat([batch['source_img'], batch['target_image']], dim=0)
                target_img =torch.cat([batch['target_image'],batch['source_image']], dim =0)
                target_pose = torch.cat([batch['taget_skeleton'],batch['source_skeleton']], dim=0)

                img = img.to(device)
                target_img = target_img.to(device)
                target_pose = target_pose.to(device)

                time_t = torch.randint(0, conf.diffusion.beta_scheduler['n_timestep'],
                                       (img.shape[0]),
                                       device=device)
                loss_dict = diffusion.training_losses(model,
                                                      x_start =target_img,
                                                      t = time_t,
                                                      cond_input = [img, target_pose],
                                                      prob = 1 - guidance_prob)
                loss     = loss_dict['loss'].mean()
                loss_mse = loss_dict['mse'].mean()
                loss_vb  = loss_dict['vb'].mean()

                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 1)
                scheduler.step()
                optimizer.step()

                loss_list.append(loss.detach().item())
                loss_mean_list.append(loss_mse.detach().item())
                loss_vb_list.append(loss_vb.detach().item())

                accumulate(ema, model.module, 0 if i < conf.training.scheduler.warmup else 0.9999)

                if i % args.save_wandb_logs_every_iters == 0 and is_main_process() :
                    wandb.log({'loss'      : (sum(loss_list)/len(loss_list)),
                               'loss_vb'   : (sum(loss_vb_list)/len(loss_vb_list)),
                               'loss_mean' : (sum(loss_mean_list)/len(loss_mean_list)),
                               'epoch'     : epoch,
                               'steps'     : i})
                    loss_list = []
                    loss_mean_list = []
                    loss_vb_list = []

                if i % args.save_checkpoint_every_iters == 0 and is_main_process() :
                    if conf.distributed :
                        model_module = model.module
                    else :
                        model_module = model
                    torch.save({'model'     : model_module.state_dict(),
                                'ema'       : ema.state_dict(),
                                'scheduler' : scheduler.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                                'conf'      : conf},
                               conf.training.ckptpath + f'/model_{str(i).zfill(6)}.pt')
                if (epoch)%args.save_wandb_images_every_epoochs == 0 :
                    print(f'Generating sdamples at epoch numer {epoch}')
                    val_batch = next(val_loader)
                    val_img = val_batch['source_image'].cuda()
                    val_pose = val_batch['target_skeleton'].cuda()
                    with torch.no_grad() :
                        if args.sample_algorithm == 'ddpm' :
                            print(f'Sampling algorithm used : DDPM')
                            samples = diffusion.p_sample_loop(ema, x_cond = [val_img, val_pose],
                                                              progress = True,
                                                              cond_scale = cond-scale)
                        elif args.sample_algorithm == 'ddim' :
                            print(f'Sampling algorithm used : DDIM')
                            nsteps = 50
                            noise = torch.randn(val_img.shape).cuda()
                            seq = range(0, 1000, 1000//nsteps)
                            xs, x0_pred = ddim_steps(noise, seq, ema, betas.cuda(),[val_img, val_pose])

                    grid = torch.cat([val_img, val_pose[:,:,:3], samples], dim = -1)
                    gathered_samples = [torch.zeros_list(grid) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_samples, grid)

                    if is_main_process() :
                        wandb.log({'samples' : wandb.Image(torch.cat(gathered_samples, dim=-2))})

def main(settings, EXP_name) :
    [args, DiffConf, DataConf] = settings
    if is_main_process() :
        wandb.init(project = 'person_synthesis', name = EXP_name, settings = wandb.Settings(code_dir = '.'))
    if DiffConf.ckpt is not None :
        DiffConf.training.scheduler.warmup = 0

    DiffConf.distributed = True
    local_rank = int(os.environ['LOCAL_RANK'])

if __name__ == '__main__' :
    init_distributed()
    # what means decsription?


    parser = argparse.ArgumentParser(description = 'sy practice code')
