import hashlib
from kernel_hmc.tools.log import logger

def sha1sum(fname, blocksize=65536):
    """
    Computes sha1sum of the given file. Same as the unix command line hash.
    
    Returns: string with the hex-formatted sha1sum hash
    """
    hasher = hashlib.sha1()
    with open(fname, 'rb') as afile:
        logger.debug("Hasing %s" % fname)
        buf = afile.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(blocksize)
    return hasher.hexdigest()