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

def grad_penalty(A_r,x_r,A_f,x_f,netD):
    
    eps   = torch.rand(A_r.size()[0])
    eps_x = torch.unsqueeze(torch.unsqueeze(eps,-1),-1)
    eps_A = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(eps,-1),-1),-1)
    
    x_hat = eps_x*x_r + (1-eps_x)*x_f
    A_hat = eps_A*A_r + (1-eps_A)*A_f
    
    d_hat = netD(A_hat,x_hat)
    
    gradients_x = grad(outputs=d_hat, inputs=x_hat,
                              grad_outputs=torch.ones(d_hat.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty_x = ((gradients_x.norm(2, dim=(1,2)) ** 2).mean()
                        
    gradient_a = grad(outputs=d_hat, inputs=A_hat,
                              grad_outputs=torch.ones(d_hat.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    gradient_penalty_a = ((gradients_a.norm(2, dim=(1,2,3)) ** 2).mean()
                          
    return gradient_penalty_x+gradient_penalty_a
                        
    
                        
     
                        
                        
    

