from GAN.models.DatasetHelper import DatasetHelper
from GAN.models.Discriminator import Discriminator
from GAN.models.Generator import Generator


def load_model(model):
    if type not in ['generator', 'discriminator', 'cnn']:
        raise ValueError('Given unknown parameter. Try cnn, generator or discriminator')
    if type == 'generator':
        generator = Generator()
        if link is None:
            return generator.apply(weights_init_normal)
        generator.load_state_dict(torch.load(link))
        generator.train()
        return generator
    if type == 'discriminator':
        discriminator = Discriminator()
        if link is None:
            return discriminator.apply(weights_init_normal)
        discriminator.load_state_dict(torch.load(link))
        discriminator.train()
        return discriminator