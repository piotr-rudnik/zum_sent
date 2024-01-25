from cnn import cnn
from glove import glove
from lstm import lstm

while True:
    zdanie = input("Podaj zdanie: ")
    print("INPUT: ", zdanie)
    print(lstm(zdanie))
    print(glove(zdanie))
    print(cnn(zdanie))

