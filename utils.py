import torch

real_label = 1.
fake_label = 0.

criterion = torch.nn.BCEWithLogitsLoss()


def gan_loss_dis(A_r,x_r,A_f,x_f,netD):

    b_size   = x_f.size()[0]
    label_r  = torch.full((b_size,), real_label, dtype=torch.float)
    label_f  = torch.full((b_size,), fake_label, dtype=torch.float)

    output_r  = netD(A_r,x_r).view(-1)
    Loss_real = criterion(output_r,label_r)

    output_f  = netD(A_f.detach(),x_f.detach()).view(-1)
    Loss_fake = criterion(output_f,label_f)

    return -Loss_real+Loss_fake


def gan_loss_gen(A_f,x_f,netD):

    b_size   = x_f.size()[0]
    label_r  = torch.full((b_size,), real_label, dtype=torch.float)
    output_f = netD(A_f,x_f).view(-1)
    Loss_gen = criterion(output_f,label_r)

    return -Loss_gen


def wgan_dis(A_r,x_r,A_f,x_f,netD):

  Loss_real = netD(A_r,x_r).mean()
  Loss_fake = netD(A_f.detach(),x_f.detach()).mean()

  return Loss_real-Loss_fake


def wgan_gen(A_f,x_f,netD):
  return netD(A_f,x_f).mean()



