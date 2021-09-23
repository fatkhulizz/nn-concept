import numpy as np
"""
    Untuk pembuatan neural network disini menggunakan OOP
    layer dianggap sebagai objek sehingga memudahkan 
    pembuatan layer baru dan dapat menyingkat penulisan kode
"""
class Neuron():
    """
        class neuron diinisialisasi dengan parameter input, beban dan bias
        input = sebagai masukan (input) bentuk matriks
        beban = sebagai beban (weight) bentuk matriks
        bias = sebagai bias bentuk matriks
    """
    def __init__(self, input, beban, bias):
        self.input = input
        self.beban = beban
        self.bias = bias
        """
            dilakukan operasi matriks dot untuk input dan beban
            kemudian ditambahkan dengan matriks nilai bias
            sesuai dengan persamaan untuk neural network
            n = w (beban) * x (input) + b (bias)
        """
        dot = np.dot(self.input, self.beban) 
        self.pre_act= np.add(dot, self.bias)

        """
            method deskripsi berfungsi untuk menampilkan informasi
            terkait nilai input, beban, bias dan hasil persamaan n
        """
    def deskripsi(self):
        return f"matriks input {np.shape(self.input)}:\n{self.input}\nmatriks beban {np.shape(self.beban)}:\n{self.beban}\nmatriks bias {np.shape(self.bias)}:\n{self.bias}\nnilai node sebelum diaktivasi {np.shape(self.pre_act)}:\n{self.pre_act}"

        """
            method sigmoid digunakan memasukkan hasil persamaan n
            kedalam fungsi aktivasi sigmoid
        """
    def sigmoid(self):
        self.post_act_sigmoid = 1 / (1 + np.exp(-self.pre_act))
        return self.post_act_sigmoid

        """
            method sigmoid digunakan memasukkan hasil persamaan n
            kedalam fungsi aktivasi relu
        """
    def relu(self):
        return np.abs(self.pre_act) * (self.pre_act > 0)

#LAYER PERTAMA
#deklarasi nilai matriks input
input1 = np.array([[1,1]])
#deklarasi nilai matriks beban
beban1 = np.array([[0.2, 0.1, 0.3],
                    [0.2, 0.4, 0.1]])
#deklarasi nilai matriks bias
bias1 = np.array([[0,0,0]])
#deklarasikan instance dengan class Neuron dan parameternya adalah nilai diatas
layer1 = Neuron(input1,beban1,bias1)
#menampilkan informasi terkait objek layer1
print(f"\nLAYER PERTAMA \n{layer1.deskripsi()}")
#menampilkan hasil aktivasi sigmoid untuk layer1
print(f"nilai node setelah diaktivasi dengan fungsi sigmoid {np.shape(layer1.sigmoid())}:\n{layer1.sigmoid()}")
#menampilkan hasil aktivasi sigmoid untuk layer2
print(f"nilai node setelah diaktivasi dengan fungsi relu {np.shape(layer1.relu())}:\n{layer1.relu()}")

#LAYER KEDUA
#input layer 2 dideklarasikan sebagai nilai dari hasil aktivasi function layer 1 dengan fungsi aktivasi sigmoid 
input2 = layer1.sigmoid()
#beban layer 2 dideklarasikan sebagai matriks dengan shape 3x4 yang nilainya random
beban2 = np.random.random((3,4))
#bias layer 2 dideklarasikan sebagai matriks dengan shape 1x4 yang nilainya random
bias2 = np.random.random((1,4))
#deklarasikan instance dengan class Neuron dan parameternya adalah nilai diatas
layer2 = Neuron(input2,beban2,bias2)
print(f"\nLAYER KEDUA \n{layer2.deskripsi()}")
print(f"nilai node setelah diaktivasi dengan fungsi sigmoid {np.shape(layer2.sigmoid())}:\n{layer2.sigmoid()}")
print(f"nilai node setelah diaktivasi dengan fungsi relu {np.shape(layer2.relu())}:\n{layer2.relu()}")

