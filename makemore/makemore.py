from bigram import Bigram

def main():
    names = open("names.txt", "r").read().splitlines()
    print(names[:10])

    
    model = Bigram(names)

    num_names = 3

    for _ in range(num_names):
        name = ''
        prev_char = 0
        total_loss = 0
        count = 0

        while True:
            char, loss = model(prev_char)
            if loss:
                total_loss += loss
            count += 1
            if char.item() == 0:
                break
            name += (model.itos[char.item()])
            prev_char = char.item()
        print(name)

    

if __name__ == "__main__":
    main()