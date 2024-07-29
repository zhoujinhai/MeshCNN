class Test(object):
    def __init__(self):
        self.i = 1
        print("init!")

    def modify(self):
        self.i += 1

    def do(self):
        print(self.i)
 
if __name__ == "__main__":
    test_class = Test()
    test_class.do()
    test_class.modify()
    test_class.do()
