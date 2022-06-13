
# update discriminator
pr = dis(real)
fake = gen(torch.rand((batch_size, 100), device='cuda')*2-1)
pf = dis(fake)
dis_loss = torch.mean(-pr) + torch.mean(pf)
dis_loss.backward()
disOpt.step()

# update generator
genOpt.zero_grad()
fake = gen(torch.rand((batch_size, 100), device='cuda')*2-1)
pf = dis(fake)
gen_loss = torch.mean(-pf)
gen_loss.backward()
genOpt.step()
