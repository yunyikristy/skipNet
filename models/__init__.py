from .m3d_gan import M3D


def create_model(name, hparams):
  if name == 'm3d':
    return M3D(hparams)
  else:
    raise Exception('Unknown model: ' + name)
