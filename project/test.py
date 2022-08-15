from dataloader import *
from model import *
from PIL import Image
from torchvision.utils import save_image

test_dataset = ImageDataset("./data/facades", transforms=transforms, mode="test")
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=0)
# generator and discriminator initialization
generator = GeneratorUNet()
discriminator = Discriminator()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

generator.to(device)
discriminator.to(device)

generator.load_state_dict(torch.load("best_model/Pix2Pix_Generator_for_Facades.pt", map_location=map_location))
discriminator.load_state_dict(torch.load("best_model/Pix2Pix_Discriminator_for_Facades.pt", map_location=map_location))

generator.eval();
discriminator.eval();

imgs = next(iter(test_dataloader)) # generate 10 images
real_A = imgs["B"].to(device)
real_B = imgs["A"].to(device)
fake_B = generator(real_A)
# real_A: condition, fake_B: translated image, real_B: original image
img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2) # concatenation by height
save_image(img_sample, f"result.png", nrow=5, normalize=True)