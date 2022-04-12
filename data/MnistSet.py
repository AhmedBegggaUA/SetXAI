import os
import numpy as np
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.transforms.functional as T
from torch.utils.data import DataLoader


"""
    Función encargada de devolver un objeto de tipo
    dataloader en pytorch
"""
def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )

class MNISTSet(torch.utils.data.Dataset):
    """
        Clase que contendrá todos los atributos de nuestro dataset:
            
            *threshold: Valor a partir seleccionaremos que pixeles meteremos 
                        en nuestro dataset
            
            *train:     Valor booleano que nos indicará si queremos el dataset
                        para el entreno o para el test
            
            *root:      El nombre del dataset que queremos importar desde la librería
                        de pytorch
            
            *full:      Valor booleano que nos indica si queremos todos los datos del
                        dataset
            *max:       Valor que indica el número máximo de elementos en cada set
            
            *data:      Contiene todos los sets que hemos generado

    """
    def __init__(self, threshold=0.0, train=True, root="mnist", full=False):
        self.train = train
        self.root = './'
        self.threshold = threshold
        self.full = full
        #Common transformations in pytorch
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        #Import mnist dataset from pytorch
        mnist = torchvision.datasets.MNIST(
            train=train, transform=transform, download=True, root=root
        )
        self.data = self.cache(mnist)
        self.max = 342
        #Función encargada de comprobar si existe alguna ejecucción previa que haya guardado
        # el dataset, de esta forma no tendriamos que procesarlo cada vez que lo usemos.
        # En caso de que no exista dicho archivo, pasaremos a crear el dataset, pasando de 
        # imagenes a set de pixeles guardados según si superán o no el threshold 
    def cache(self, dataset):
        cache_path = os.path.join(self.root, f"mnist_{self.train}_{self.threshold}.pth")
        if os.path.exists(cache_path):
            return torch.load(cache_path)
    
        print("Procesando el dataset...")
        data = []
        for datapoint in dataset:
            img, label = datapoint
            point_set, cardinality = self.image_to_set(img)
            data.append((point_set, label, cardinality))
        torch.save(data, cache_path)
        print("Listo!")
        return data

        # Función que recibe como parametro una imagen de mnist la cual
        # tiene una dimesión de 1x28x28, es por ello que primero se le elimina
        # la primera dimesión y acto seguido comprobamos cual de los 28x28 pixeles
        # que tenemos supera el threshold. Nos quedamos con las coordenadas de dichas posiciones
        # con la función nonzero(). Finalmente sacamos la cardinalidad, que no es mas que el número
        # de puntos totales que han superado el threshold

    def image_to_set(self, img):
        idx = (img.squeeze(0) > self.threshold).nonzero().transpose(0, 1)
        cardinality = idx.size(1)
        return idx, cardinality
        
        # Función que devuelve un elemento del dataset
    def __getitem__(self, item):
        # s: tensor de dos dimensiones que contiene coordenadas de los pixeles que superán el threshold
        # l: etiqueta correspondiente a la imagen que hemos procesado
        # c: número de pixeles que superán el threshold  
        s, l, c = self.data[item]
        # make sure set is shuffled
        s = s[:, torch.randperm(c)]
        # pad to fixed size, esto se hace para que todos los sets tengan un tamaño de max, rellenando con 0s
        # las posiciones que faltan
        padding_size = self.max - s.size(1)
        s = torch.cat([s.float(), torch.zeros(2, padding_size)], dim=1)
        # put in range [0, 1]
        s = s / 27
        # creamos un tensor que llamaremos mascará, que su principal función es decirnos que elementos del set
        # son relleno y cuales no
        mask = torch.zeros(self.max)
        mask[:c].fill_(1)
        return l, s, mask

        # Función que devuelve la longitud de nuestro dataset, necesaría para el loader de pytorch
    def __len__(self):
        if self.train or self.full:
            return len(self.data)
        else:
            return len(self.data) // 10

#if __name__ == "__main__":
#    dataset = MNISTSet(full=True)
#    train_loader = get_loader(dataset,32)
#    print(train_loader)