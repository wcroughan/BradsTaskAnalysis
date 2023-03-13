import time
import multiprocessing


class C:
    def __init__(self, v):
        self.value = v
        self.obj = [1, 2, 3]
        self.s = "s"

    def setVal(self, v):
        print(self.value)
        self.value = v
        print("set")

    def p(self):
        print(self.s)
        self.obj.append(4)
        self.s = ",".join([str(i) for i in self.obj])
        print(self.s)
        print("pushed")


def f(c, i):
    time.sleep(i / 10)
    c.p()


def main():
    c = C(0)
    with multiprocessing.Pool() as pool:
        pool.starmap(f, zip([c] * 10, range(5, 10)))
    print(c.value)


if __name__ == "__main__":
    main()
