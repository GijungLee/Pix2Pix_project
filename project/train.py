from torchvision.utils import save_image
from dataloader import *
from model import *

train_dataset = ImageDataset("./data/facades", transforms=transforms, mode="train")
test_dataset = ImageDataset("./data/facades", transforms=transforms, mode="test")

train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=0)

generator = GeneratorUNet()
discriminator = Discriminator()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

generator.to(device)
discriminator.to(device)

# weights initialization
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# loss function
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

criterion_GAN.to(device)
criterion_pixelwise.to(device)

# learning rate
lr = 0.0002

# Optimizer for generator and discriminator
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

import time

n_epochs = 200 # epoch
sample_interval = 200 # how much time a result will be showed

# weight parameter for L1 pixel-wise between translated image and original image
lambda_pixel = 100

start_time = time.time()

for epoch in range(n_epochs):
    for i, batch in enumerate(train_dataloader):
        # input data for model
        real_A = batch["B"].to(device)
        real_B = batch["A"].to(device)

        # generate target label for real and fake images width and height are divided by 16
        real = torch.FloatTensor(real_A.size(0), 1, 16, 16).fill_(1.0).to(device)  # real: 1
        fake = torch.FloatTensor(real_A.size(0), 1, 16, 16).fill_(0.0).to(device)  # fake: 0

        """ training generator"""
        optimizer_G.zero_grad()

        # Generate image
        fake_B = generator(real_A)

        # calculate the loss for generator
        loss_GAN = criterion_GAN(discriminator(fake_B, real_A), real)

        # calculate the loss for pixel-wise L1
        loss_pixel = criterion_pixelwise(fake_B, real_B)

        # total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        # updata generator
        loss_G.backward()
        optimizer_G.step()

        """ training discriminator """
        optimizer_D.zero_grad()

        # calculate the loss for discriminator
        loss_real = criterion_GAN(discriminator(real_B, real_A), real) # 조건(condition): real_A
        loss_fake = criterion_GAN(discriminator(fake_B.detach(), real_A), fake)
        loss_D = (loss_real + loss_fake) / 2

        # update discriminator
        loss_D.backward()
        optimizer_D.step()

        done = epoch * len(train_dataloader) + i
        if done % sample_interval == 0:
            imgs = next(iter(test_dataloader)) # generate 10 images
            real_A = imgs["B"].to(device)
            real_B = imgs["A"].to(device)
            fake_B = generator(real_A)
            # real_A: condition, fake_B: translated image, real_B: original image
            img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2) # concatenation by height
            save_image(img_sample, f"{done}.png", nrow=5, normalize=True)

    # print log every epoch
    print(f"[Epoch {epoch}/{n_epochs}] [D loss: {loss_D.item():.6f}] [G pixel loss: {loss_pixel.item():.6f}, adv loss: {loss_GAN.item()}] [Elapsed time: {time.time() - start_time:.2f}s]")

# save model
torch.save(generator.state_dict(), "Pix2Pix_Generator_for_Facades.pt")
torch.save(discriminator.state_dict(), "Pix2Pix_Discriminator_for_Facades.pt")
print("Model saved!")