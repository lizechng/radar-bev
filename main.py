from fire import Fire

import src

if __name__ == '__main__':
    Fire({
        'train': src.train.train,
        'train_gan': src.train_gan.train,
        'model_test': src.test.model_test,
        'knn_test': src.test_knn.knn_test,
    })