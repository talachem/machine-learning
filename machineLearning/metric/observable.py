from ..utility.format import Format


class Observable(object):
    """
    This class is used during training of neural network
    keeps track of a single training parameter
    """
    __slots__ = ['name', 'values', 'epochs', '__index__', 'colors', 'ff']

    def __init__(self, epochs: int, colors: str = None) -> None:
        self.name = self.__class__.__name__
        self.values = [0] * epochs
        self.epochs = epochs
        self.__index__ = 0
        self.colors = colors
        self.ff = Format()

    def update(self, value: float, scale: int | float = 1.) -> None:
        """
        Updates the current value with a scaling factor.
        """
        self.values[self.__index__] += value / scale

    def step(self) -> None:
        """
        Increases the index at which values will be added by one.
        This should be used at the end of each epoch to step through the observable.
        """
        self.__index__ += 1

    def print(self) -> str:
        """
        Prepares the current value for printing during training.
        It color codes the print out in accordance with decrease and increase.
        """
        value = self.values[self.__index__]
        color = 'white'
        if self.__index__ > 1 and (self.colors == 'descending' or self.colors == 'ascending'):
            comparate = (self.values[self.__index__-2] + self.values[self.__index__-1]) / 2
            bestValue = max(self.values[2:]) if self.colors == 'ascending' else min([item for item in self.values[2:] if item > 0])
            if (bestValue >= value):
                color = 'green' if self.colors == 'descending' else 'red'
            elif (comparate > value) and (bestValue < value):
                color = 'yellow' if self.colors == 'descending' else 'red'
            elif (value >= bestValue):
                color = 'green' if self.colors == 'ascending' else 'red'
            elif (value > comparate) and (value < bestValue):
                color = 'yellow' if self.colors == 'ascending' else 'red'
            else:
                color = 'white'
        return self.ff(str(round(value, 5)), color=color, style='bold')


class Observables(object):
    """
    Stores all the observables of the training process.
    It tracks the training and validation loss during training and writes out the learning rate.
    Additionally, it prints/writes the losses to CLI/a log file and plots the losses after training.
    """

    def __init__(self, epochs: int) -> None:
        self.name = self.__class__.__name__
        self.epochs = epochs
        self._metrics = [] # this stores variables for print out

    def step(self) -> None:
        """
        Steps through epochs for each observable.
        """
        for observable in self._metrics:
            getattr(self, observable).step()

    def addOberservable(self, name: str, colors: str = None) -> None:
        """
        add parameters/metrics one wants to observe during training
        """
        setattr(self, name, Observable(self.epochs, colors))

        # used for accessing observables for print out
        # preserves the order in which the user adds the observable
        self._metrics.append(name)

    def update(self, observable: str, *args: float) -> None:
        """
        Updates the observable, it can be scaled
        """
        getattr(self, observable).update(*args)

    def __str__(self) -> str:
        """
        Returns the string representation of the current observables during the training run.
        """
        string = ''
        for attr in self._metrics:
            string += attr + ": "
            string += getattr(self, attr).print().ljust(21) + "   "
        return string

    def print(self) -> None:
        """
        Prints out the current observables during the training run.
        """
        print(self)

    def writeOut(self, runName: str) -> None:
        """
        Writes out the values of the run into a log file.
        These can be used for replotting the loss curves.
        """
        g = open(f"outputs/{runName}-training.txt", "w")
        header = ' training '.center(95, '–')
        g.write(f"{header}\n")
        heading = 'epoch' + "    "
        for item in self._metrics:
            heading += str(item).ljust(18) + "    "
        heading += '\n'
        g.write(f"{heading}")
        epochs = len(self.losses.values)
        leadingZeros = len(str(epochs))
        for i in range(epochs):
            now = str(i+1)
            string = now.zfill(leadingZeros).ljust(len('epoch')) + "    "
            for attr in self._metrics:
                printValue = round(abs(getattr(self, attr).values[i]),16)
                string += str(printValue).ljust(18) + "    "
            string += '\n'
            g.write(f"{string}")
        g.close()
