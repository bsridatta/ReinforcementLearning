import numpy as np
import matplotlib.pyplot as plt

class Maze():

    def __init__(self, layout, A, B):
        self.maze = layout
        self.A = A
        self.B = B

    def draw_maze(self):
     
        fig, ax = plt.subplots()

        plt.grid(True)
        plt.tick_params(axis='both', labelsize=0, length = 0)
        ax.set_xticks(np.arange(0.5, self.maze.shape[0], 1))
        ax.set_yticks(np.arange(0.5, self.maze.shape[1], 1))

        ax.text(self.A[1], self.A[0], "A", horizontalalignment='center', verticalalignment='center')
        ax.text(self.B[1], self.B[0], "B", horizontalalignment='center', verticalalignment='center')

        plt.imshow(self.maze, cmap='gray',  interpolation="none")
        plt.show()
        plt.close()

  

if __name__ == '__main__':

    layout =  [
       [ 0,  0,  1,  0,  0,  0,  0,  0],
       [ 0,  0,  1,  0,  0,  1,  0,  0],
       [ 0,  0,  1,  0,  0,  1,  1,  1],
       [ 0,  0,  1,  0,  0,  1,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  1,  1,  1,  1,  1,  1,  0],
       [ 0,  0,  0,  0,  1,  0,  0,  0]]

    A = [0, 0]
    B = [6, 5]

    maze = Maze(np.asarray(layout), A, B)
    maze.draw_maze()
    print("main")