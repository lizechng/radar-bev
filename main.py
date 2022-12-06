from fire import Fire

import src

if __name_ == '__main__':
    Fire({
        'train': src.train.train,
        'train_gan': src.train_gan.train,
        'model_test': src.test.model_test
    })