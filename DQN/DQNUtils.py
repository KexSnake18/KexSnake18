def yes_no_question(question, yes_print=None, no_print=None):
    print(question)
    while True:
        inp = input("Choose (y/n): ")
        if inp == "y":
            if yes_print: print(yes_print)
            return True
        elif inp == "n":
            if no_print: print(no_print)
        else:
            print("Choose y/n")
                

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])